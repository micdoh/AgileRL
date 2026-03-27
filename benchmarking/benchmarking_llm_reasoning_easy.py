from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
    raise ImportError(
        msg,
    )

import re

import yaml
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from agilerl.algorithms import LLMPPO
from agilerl.training.train_llm import finetune_llm_reasoning
from agilerl.utils.llm_utils import ReasoningGym
from agilerl.utils.algo_utils import VLLMConfig

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"
MAX_CONTEXT_LENGTH = 1024
USE_TINY_DEBUG_MODEL = False
USE_VLLM = not USE_TINY_DEBUG_MODEL


def make_dataset(dataset_name: str) -> tuple[Dataset, Dataset]:
    raw_dataset = (
        load_dataset(dataset_name, split="train").shuffle(seed=42).select(range(50000))
    )
    raw_dataset = raw_dataset.rename_column("target", "answer")
    raw_dataset = raw_dataset.rename_column("nums", "question")
    train_test_split = raw_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    return train_dataset, test_dataset


def reward_fn(completion, answer, question):
    """Reward for adding numbers together with <reasoning>/<answer> format.

    The 'answer' field from the dataset is ignored -- the correct answer is
    simply sum(question).

    Format (0.1 each, 0.5 total):
      1. Contains <reasoning> opening tag
      2. Contains </reasoning> closing tag
      3. Contains <answer> opening tag
      4. Contains </answer> closing tag
      5. Correct ordering (reasoning closes before answer opens)

    Correctness (0.5):
      The number inside <answer>...</answer> equals sum(question)

    Perfect format bonus (0.5):
      Full response matches <reasoning>...</reasoning>\n<answer>...</answer>
      with non-empty content in both sections.
    """
    target = sum(question)
    reward = 0.0

    has_reason_open = "<reasoning>" in completion
    has_reason_close = "</reasoning>" in completion
    has_answer_open = "<answer>" in completion
    has_answer_close = "</answer>" in completion

    if has_reason_open:
        reward += 0.1
    if has_reason_close:
        reward += 0.1
    if has_answer_open:
        reward += 0.1
    if has_answer_close:
        reward += 0.1

    if has_reason_open and has_reason_close and has_answer_open and has_answer_close:
        reason_close_idx = completion.index("</reasoning>")
        answer_open_idx = completion.index("<answer>")
        if reason_close_idx < answer_open_idx:
            reward += 0.1

            answer_content = completion[
                completion.index("<answer>") + len("<answer>"):
                completion.index("</answer>")
            ].strip()
            numbers = re.findall(r"-?\d+", answer_content)
            if numbers and int(numbers[-1]) == target:
                reward += 0.5

            reasoning_content = completion[
                completion.index("<reasoning>") + len("<reasoning>"):
                reason_close_idx
            ].strip()
            if reasoning_content and answer_content:
                match = re.match(
                    r"^\s*<reasoning>.+</reasoning>\s*<answer>.+</answer>\s*$",
                    completion,
                    re.DOTALL,
                )
                if match:
                    reward += 0.5

    return reward


def main(init_hp, mut_p):

    if USE_TINY_DEBUG_MODEL:
        from benchmarking.tiny_model import build_tiny_actor_network, TinyDigitTokenizer

        actor_network = build_tiny_actor_network()
        tokenizer = TinyDigitTokenizer()
        model_name = None
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        actor_network = None
        model_name = MODEL_PATH
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]

    print("Tokenizer", tokenizer.pad_token, tokenizer.eos_token)
    train_dataset, test_dataset = make_dataset(DATASET)

    conversation_template = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Show your reasoning in <reasoning> </reasoning> tags, then give your final answer in <answer> </answer> tags.",
        },
        {
            "role": "user",
            "content": "What is the sum of the following numbers: {question}?",
        },
        {"role": "assistant", "content": ""},
    ]

    accelerator = Accelerator() if not USE_TINY_DEBUG_MODEL else None
    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        conversation_template=conversation_template,
        data_batch_size_per_gpu=init_hp["BATCH_SIZE"],
        accelerator=accelerator,
        max_context_length=MAX_CONTEXT_LENGTH,
        return_raw_completions=USE_VLLM,
    )

    init_hp["ALGO"] = "LLMPPO"
    init_hp["MAX_MODEL_LEN"] = MAX_CONTEXT_LENGTH
    print("pad token id", tokenizer.pad_token_id)
    assert tokenizer.pad_token_id != tokenizer.eos_token_id, (
        "Pad token and eos token are the same"
    )

    llm_ppo = LLMPPO(
        model_name=model_name,
        actor_network=actor_network,
        lora_config=LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        micro_batch_size_per_gpu=8
        if init_hp["BATCH_SIZE"] > 8
        else init_hp["BATCH_SIZE"],
        use_vllm=USE_VLLM,
        pad_token_id=tokenizer.pad_token_id,
        pad_token=tokenizer.pad_token,
        use_separate_reference_adapter=True,
        batch_size=init_hp["BATCH_SIZE"],
        beta=init_hp["BETA"],
        lr=init_hp["LR"],
        clip_coef=init_hp["CLIP_COEF"],
        max_grad_norm=init_hp["MAX_GRAD_NORM"],
        update_epochs=init_hp["UPDATE_EPOCHS"],
        temperature=init_hp["TEMPERATURE"],
        max_model_len=init_hp["MAX_MODEL_LEN"],
        accelerator=accelerator,
        vf_coef=init_hp["VF_COEF"],
        gamma=init_hp["GAMMA"],
        gae_lambda=init_hp["GAE_LAMBDA"],
        vllm_config=VLLMConfig(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_num_seqs=2,
            sleep_mode=True,
        ),
    )

    print("llm_ppo.lr", llm_ppo.lr)

    finetune_llm_reasoning(
        pop=[llm_ppo],
        env=env,
        init_hp=init_hp,
        evaluation_interval=10,
        wb=True,
        save_elite=True,
        elite_path="saved_llms",
        max_reward=1.5,
        evo_steps=None,
        mutation=None,
        tournament=None,
        accelerator=accelerator,
        verbose=True,
    )
    accelerator.end_training()


if __name__ == "__main__":
    with open("configs/training/llm_finetuning/ppo_llm.yaml") as file:
        config = yaml.safe_load(file)
        print(config)
    init_hp = config["INIT_HP"]
    mut_p = config["MUTATION_PARAMS"]
    main(init_hp, mut_p)
