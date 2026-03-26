import gc
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from torch.nn.utils import clip_grad_norm_
import dataclasses
from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core import LLMAlgorithm, OptimizerWrapper
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.modules.dummy import DummyEvolvable
from agilerl.protocols import (
    LoraConfigProtocol,
    PeftModelProtocol,
    PreTrainedModelProtocol,
)
from agilerl.typing import ExperiencesType, LLMObsType
from agilerl.utils.algo_utils import (
    CosineLRScheduleConfig,
    DummyOptimizer,
    VLLMConfig,
    create_warmup_cosine_scheduler,
    get_experiences_samples,
    stack_and_pad_experiences,
)
from agilerl.utils.llm_utils import (
    ReasoningGym,
    masked_mean,
    masked_whiten,
)

if HAS_LLM_DEPENDENCIES:
    from transformers import GenerationConfig


class PPO(LLMAlgorithm):
    """Token-level PPO for LLM finetuning using a single model with actor/reference/critic adapters.

    Unlike ppo_llm.py (which uses two separate model instances), this implementation uses a single
    AutoModelForCausalLMWithValueHead backbone with three named LoRA adapters: "actor", "reference",
    and "critic". Adapter switching is done via select_adapter(); _restore_lora_trainability() is
    called before each backward pass to ensure ZeRO-2 gradient hooks fire correctly.
    """

    def __init__(
        self,
        pad_token_id: int,
        pad_token: str,
        model_name: str | None = None,
        actor_network: Any | None = None,
        model_config: dict[str, Any] | None = None,
        hp_config: HyperparameterConfig | None = None,
        index: int = 0,
        batch_size: int = 16,
        beta: float = 0.01,
        vf_coef: float = 0.5,
        clip_coef: float = 0.2,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
        lr: float = 5e-7,
        max_grad_norm: float = 1.0,
        update_epochs: int = 1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        min_p: float = 0.0,
        use_separate_reference_adapter: bool = True,
        calc_position_embeddings: bool = True,
        micro_batch_size_per_gpu: int | None = None,
        reduce_memory_peak: bool = False,
        max_output_tokens: int | None = 1024,
        min_output_tokens: int | None = None,
        max_model_len: int | None = None,
        lora_config: LoraConfigProtocol | None = None,
        cosine_lr_schedule_config: CosineLRScheduleConfig | None = None,
        accelerator: Accelerator | None = None,
        device: str = "cpu",
        wrap: bool = True,
        clone: bool = False,
        use_vllm: bool = False,
        vllm_config: VLLMConfig | None = None,
        seed: int = 42,
        gradient_checkpointing: bool = True,
    ) -> None:

        device = (
            f"cuda:{accelerator.process_index}"
            if accelerator is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        super().__init__(
            index=index,
            batch_size=batch_size,
            lr=lr,
            max_grad_norm=max_grad_norm,
            clone=clone,
            reduce_memory_peak=reduce_memory_peak,
            calc_position_embeddings=calc_position_embeddings,
            seed=seed,
            pad_token_id=pad_token_id,
            pad_token=pad_token,
            use_value_head=True,  # single model with value head; critic adapter added by _initialize_actors
            use_liger_loss=False,
            lora_config=lora_config,
            use_separate_reference_adapter=use_separate_reference_adapter,
            model_name=model_name,
            actor_network=actor_network,
            model_config=model_config,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            cosine_lr_schedule_config=cosine_lr_schedule_config,
            hp_config=hp_config,
            wrap=wrap,
            device=device,
            accelerator=accelerator,
            name="LLMPPO",
            gradient_checkpointing=gradient_checkpointing,
        )
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(clip_coef, (float, int)), "Clipping coefficient must be a float."
        assert clip_coef >= 0, (
            "Clipping coefficient must be greater than or equal to zero."
        )
        assert isinstance(update_epochs, int), "Policy update epochs must be an integer."
        assert update_epochs >= 1, (
            "Policy update epochs must be greater than or equal to one."
        )
        if clone and actor_network is not None:
            assert isinstance(
                actor_network,
                (PeftModelProtocol, PreTrainedModelProtocol),
            ), "Actor network must be a PeftModelProtocol or PreTrainedModelProtocol"
        if max_output_tokens is None and max_model_len is None:
            msg = "Either max_output_tokens or max_model_len must be specified"
            raise ValueError(msg)

        self.beta = beta
        self.vf_coef = vf_coef
        self.clip_coef = clip_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_output_tokens = max_output_tokens
        self.min_output_tokens = min_output_tokens
        self.max_model_len = (
            max_model_len if max_model_len is not None else max_output_tokens + 512
        )
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            max_length=self.max_model_len,
            max_new_tokens=max_output_tokens,
            min_new_tokens=min_output_tokens,
            pad_token_id=pad_token_id,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )

        self.use_vllm = use_vllm
        self.vllm_config = vllm_config
        if self.use_vllm:
            self._configure_vllm()
        self._initialize_actors(actor_network, not clone)
        self._initialize_critic()
        # Only actor registered — critic lives as an adapter on self.actor
        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        if self.wrap:
            self.wrap_models()

    def get_action(
        self,
        obs: LLMObsType,
        training: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return generated completion ids and corresponding action masks."""
        self.actor.eval()
        if not self.use_vllm:
            actor_module = self._get_unwrapped_actor()
            try:
                actor_device = next(actor_module.parameters()).device
            except StopIteration:
                actor_device = torch.device(self.device)
            with torch.no_grad():
                completion_ids = []
                action_masks = []
                for prompt in obs:
                    prompt.pop("text", None)
                    prompt["input_ids"] = prompt["input_ids"].to(actor_device)
                    prompt["attention_mask"] = prompt["attention_mask"].to(actor_device)
                    completion_id = self.actor.generate(
                        **prompt,
                        generation_config=self.generation_config,
                    )
                    completion_ids.append(completion_id)
                    action_mask = torch.zeros_like(
                        completion_id,
                        dtype=torch.bool,
                        device=completion_id.device,
                    )
                    action_mask[:, prompt["input_ids"].shape[1] :] = True
                    action_mask[completion_id == self.pad_token_id] = False
                    action_mask = action_mask[:, 1:]
                    action_masks.append(action_mask)
        else:
            if self.vllm_config.sleep_mode:
                torch.cuda.empty_cache()
                self.llm.wake_up()
            self._move_model_to_vllm()
            completion_ids, action_masks = self._generate_with_vllm_colocate(obs, 1)
            if self.vllm_config.sleep_mode:
                self.llm.sleep(level=2)

        return completion_ids, action_masks

    def _initialize_critic(self) -> None:
        """Register critic LoRA params in the shared optimizer as a second param group.

        The 'critic' adapter was already created by _initialize_actors (use_value_head=True).
        This makes those params visible to the optimizer so a single backward pass updates
        both actor and critic. Note: requires_grad may be False here because set_adapter("actor")
        was called at the end of _initialize_actors — _restore_lora_trainability re-enables
        them before each backward.
        """
        config = dataclasses.replace(self.lora_config)
        config.target_modules.add("summary")
        self.actor.add_adapter(
            adapter_name="critic",
            peft_config=config,  # type: ignore[arg-type]
        )
        critic_lora_params = [
            p for n, p in self.actor.named_parameters()
            if "critic" in n and "lora" in n
        ]
        self.optimizer.optimizer.add_param_group({"params": critic_lora_params, "lr": self.lr})

    def _backward_pass(self, loss: torch.Tensor) -> None:
        """Single backward pass updating both actor and critic via the shared optimizer.

        Calls _restore_lora_trainability before backward because select_adapter() calls
        set_adapter() which toggles requires_grad=False on non-active adapters. This is
        critical for ZeRO-2 where gradient reduction hooks are registered once at
        accelerator.prepare() time and require requires_grad=True to fire correctly.
        """
        self._restore_lora_trainability(["actor", "critic"])
        if self.accelerator is not None:
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            loss.backward()
            all_lora_params = [
                p for group in self.optimizer.optimizer.param_groups
                for p in group["params"]
            ]
            clip_grad_norm_(all_lora_params, self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.lr = self.lr_scheduler.get_last_lr()[0]

    def learn(
        self,
        experiences: ExperiencesType,
        tokenizer,
    ) -> tuple[float, float, float, float, float]:
        """Update actor and critic adapters using token-level PPO objectives."""
        gc.collect()
        torch.cuda.empty_cache()

        completion_ids, action_masks, rewards = stack_and_pad_experiences(
            *experiences,
            padding_values=[self.pad_token_id, False, None],
        )
        completion_ids = completion_ids.to(self.device)
        sequence_rewards = rewards.flatten().to(self.device).float()
        action_masks = action_masks.to(self.device)
        num_samples = completion_ids.shape[0]

        if sequence_rewards.shape[0] != num_samples:
            msg = (
                "Expected one scalar reward per sampled completion. "
                f"Got {sequence_rewards.shape[0]} rewards for {num_samples} samples."
            )
            raise ValueError(msg)

        batch_idxs = np.arange(num_samples)
        batch_size = (
            min(num_samples, self.micro_batch_size_per_gpu)
            if hasattr(self, "micro_batch_size_per_gpu")
            else num_samples
        )
        mean_pg_loss, mean_vf_loss, mean_loss, mean_kl, mean_entropy, updates = (
            0.0, 0.0, 0.0, 0.0, 0.0, 0,
        )

        with torch.no_grad():
            with self.select_adapter("reference"):
                reference_log_probs = self._get_logprobs(
                    completion_ids,
                    batch_size=batch_size,
                    use_reference=True,
                    eval_mode=True,
                )
            with self.select_adapter("actor"):
                old_log_probs = self._get_logprobs(
                    completion_ids,
                    batch_size=batch_size,
                    use_reference=False,
                    eval_mode=True,
                )
            with self.select_adapter("critic"):
                old_values = self._get_values(
                    completion_ids,
                    batch_size=batch_size,
                    eval_mode=True,
                )
            old_values = torch.masked_fill(old_values, ~action_masks.bool(), 0.0)

            token_rewards = self._compute_token_rewards(action_masks, sequence_rewards)
            old_log_probs = torch.masked_fill(old_log_probs, ~action_masks.bool(), 1.0)
            reference_log_probs = torch.masked_fill(reference_log_probs, ~action_masks.bool(), 1.0)
            token_kl = old_log_probs - reference_log_probs
            token_penalised_rewards = token_rewards - self.beta * token_kl
            returns, advantages = self._compute_gae_returns(token_penalised_rewards, old_values, action_masks)

        params = {}
        for name, param in self.actor.named_parameters():
            if "lora" in name and "actor" in name:
                params[name] = param.clone().detach()
        critic_params = {}
        for name, param in self.actor.named_parameters():
            if "lora" in name and "critic" in name:
                critic_params[name] = param.clone().detach()

        for _ in range(self.update_epochs):
            self.rng.shuffle(batch_idxs)
            for start in range(0, num_samples, batch_size):
                minibatch_idxs = batch_idxs[start : min((start + batch_size), num_samples)]
                (
                    batch_ids,
                    batch_action_mask,
                    batch_old_log_probs,
                    batch_reference_log_probs,
                    batch_returns,
                    batch_advantages,
                    batch_old_values,
                ) = get_experiences_samples(
                    minibatch_idxs,
                    completion_ids,
                    action_masks,
                    old_log_probs,
                    reference_log_probs,
                    returns,
                    advantages,
                    old_values,
                )

                with self.select_adapter("actor"):
                    batch_log_probs = self._get_logprobs(
                        batch_ids,
                        batch_size=batch_size,
                        use_reference=False,
                        eval_mode=False,
                    )
                    batch_log_probs = torch.masked_fill(batch_log_probs, ~batch_action_mask.bool(), 1.0)
                    kl = batch_log_probs - batch_reference_log_probs
                    masked_entropy = masked_mean(-batch_log_probs.detach(), batch_action_mask)
                    policy_ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                    clipped_ratio = torch.clamp(policy_ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss_unclipped = -batch_advantages * policy_ratio
                    pg_loss_clipped = -batch_advantages * clipped_ratio
                    pg_loss = masked_mean(torch.max(pg_loss_unclipped, pg_loss_clipped), batch_action_mask)

                with self.select_adapter("critic"):
                    batch_values = self._get_values(
                        batch_ids,
                        batch_size=batch_size,
                        eval_mode=False,
                    )
                batch_values = torch.masked_fill(batch_values, ~batch_action_mask.bool(), 0.0)
                vf_loss = (batch_returns - batch_values).pow(2)
                clipped_batch_values = batch_old_values + torch.clamp(
                    batch_values - batch_old_values,
                    -self.clip_coef,
                    self.clip_coef,
                )
                clipped_vf_loss = (batch_returns - clipped_batch_values).pow(2)
                vf_loss = 0.5 * masked_mean(torch.max(vf_loss, clipped_vf_loss), batch_action_mask) * self.vf_coef

                total_loss = pg_loss + vf_loss
                self._backward_pass(total_loss)

                mean_kl += masked_mean(kl, batch_action_mask).item()
                mean_entropy += masked_entropy.mean().item()
                del masked_entropy
                mean_pg_loss += pg_loss.mean().item()
                mean_vf_loss += vf_loss.mean().item()
                mean_loss += total_loss.item()
                updates += 1

        for name, param in self.actor.named_parameters():
            if "lora" in name and "actor" in name:
                if torch.equal(param.data, params[name].data):
                    print(f"Actor lora {name} has not updated {self.learn_ticker}")
        for name, param in self.actor.named_parameters():
            if "lora" in name and "critic" in name:
                if torch.equal(param.data, critic_params[name].data):
                    print(f"Critic lora {name} has not updated {self.learn_ticker}")

        return (
            mean_loss / max(updates, 1),
            mean_kl / max(updates, 1),
            mean_pg_loss / max(updates, 1),
            mean_vf_loss / max(updates, 1),
            mean_entropy / max(updates, 1),
        )

    def test(
        self,
        env: ReasoningGym,
        loop: int = 1,
    ) -> torch.Tensor:
        """Return fitness (test) score tensor of llm on test sub-set."""
        with env.eval_mode(), torch.no_grad():
            prompts = env.reset()
            rewards = []
            for _ in range(loop):
                completion_ids, _ = self.get_action(prompts, training=False)
                next_prompts, reward = env.step(completion_ids)
                prompts = next_prompts
                rewards.append(reward)
        reward_tensor = torch.cat(rewards)
        mean_fit = torch.mean(reward_tensor).item()
        self.fitness.append(mean_fit)
        return reward_tensor

    def _compute_gae_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE returns R_t = V_t + A_t^GAE for each token position."""
        batch_size, sequence_length = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(batch_size, device=rewards.device)

        for t in reversed(range(sequence_length)):
            mask_t = action_mask[:, t]
            if t == sequence_length - 1:
                next_values = torch.zeros_like(values[:, 0])
            else:
                next_values = values[:, t + 1] * action_mask[:, t + 1]
            delta = rewards[:, t] + self.gamma * next_values - values[:, t]
            last_gae = (delta + self.gamma * self.gae_lambda * last_gae) * mask_t
            advantages[:, t] = last_gae

        returns = advantages + values
        advantages = masked_whiten(advantages, action_mask)
        return returns, advantages * action_mask

    def _compute_token_rewards(
        self,
        action_mask: torch.Tensor,
        sequence_rewards: torch.Tensor,
    ) -> torch.Tensor:
        token_rewards = torch.zeros_like(action_mask, dtype=torch.float32)
        valid = action_mask.any(dim=-1)
        if valid.any():
            reward_idx = action_mask[valid].long().cumsum(dim=-1).argmax(dim=-1)
            row_ids = torch.arange(
                token_rewards.shape[0],
                device=token_rewards.device,
            )[valid]
            token_rewards[row_ids, reward_idx] = sequence_rewards[valid]
        return token_rewards

    def _get_values(
        self,
        ids: torch.Tensor,
        batch_size: int,
        eval_mode: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute value estimates using self.actor with the critic adapter active.

        Must be called inside a select_adapter("critic") context.
        """
        self.actor.train(mode=not eval_mode)
        num_samples = ids.shape[0]
        if attention_mask is None:
            attention_mask = ids != self.pad_token_id
        if self.calc_position_embeddings:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
        values = []
        for batch in range(0, num_samples, batch_size):
            end_idx = min((batch + batch_size), num_samples)
            batch_ids = ids[batch:end_idx, :]
            batch_attention_mask = attention_mask[batch:end_idx, :]
            batch_model_kwargs = {
                "input_ids": batch_ids,
                "attention_mask": batch_attention_mask,
                "use_cache": False,
            }
            if self.calc_position_embeddings:
                batch_position_ids = position_ids[batch:end_idx, :]
                batch_model_kwargs |= {"position_ids": batch_position_ids}
            *_, value = self.actor.forward(**batch_model_kwargs)
            values.append(value[:, :-1])
        return torch.cat(values, dim=0)

    def _get_unwrapped_actor(self) -> Any:
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self.actor)
        return self.actor
