# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation
```bash
pip install -e .          # basic install
pip install -e ".[all]"   # with LLM dependencies (transformers, peft, vllm, etc.)
```

### Testing
```bash
# Run all tests
PYTHONHASHSEED=0 pytest

# Run a single test file
PYTHONHASHSEED=0 pytest tests/test_algorithms/test_ppo.py

# Run a single test
PYTHONHASHSEED=0 pytest tests/test_algorithms/test_ppo.py::TestPPO::test_learn

# Skip LLM tests (slow, requires GPU/LLM deps)
PYTHONHASHSEED=0 pytest -m 'not llm'

# With coverage
PYTHONHASHSEED=0 pytest --exitfirst --cov=agilerl --cov-report=xml
```

### Linting
```bash
ruff check agilerl/       # lint
ruff format agilerl/      # format
pre-commit run --all-files  # run all pre-commit hooks
```

## Architecture Overview

AgileRL is a Deep RL library built around **Evolutionary Hyperparameter Optimization (HPO)**. The core idea: instead of running separate hyperparameter searches, maintain a *population* of agents during training and use evolutionary algorithms (tournament selection + mutations) to automatically converge on optimal hyperparameters.

### Key Abstraction Layers

**Algorithms** (`agilerl/algorithms/`)
- All algorithms inherit from `RLAlgorithm` or `MultiAgentRLAlgorithm` (in `core/base.py`)
- The base class provides: checkpointing, CUDA graph support, distributed training (DeepSpeed/Accelerate), and the **registry system** for evolvable hyperparameters
- Each algorithm declares which of its hyperparameters are evolvable via `HyperparameterConfig` and `NetworkGroup` objects in the `register_*` methods

**Registry System** (`agilerl/algorithms/core/registry.py`)
- Algorithms declare mutable hyperparameters using `RLParameter` (with min/max bounds, growth/shrink factors) and `NetworkGroup` (which networks evolve together)
- The `RegistryMeta` metaclass automatically initializes this registry on class creation
- This is what allows `Mutations` to know *which* hyperparameters to mutate and *how*

**Evolvable Networks/Modules** (`agilerl/networks/`, `agilerl/modules/`)
- `EvolvableModule` (`modules/base.py`) is the base for all mutable neural components — it supports architecture mutations (add/remove layers, change sizes, swap activations)
- `EvolvableNetwork` (`networks/base.py`) composes modules into encoder-decoder structures
- Algorithms hold references to networks; when the mutation engine calls `mutate_network()`, it rebuilds the network with modified architecture and copies weights where possible

**HPO Engine** (`agilerl/hpo/`)
- `Mutations` (`hpo/mutation.py`): Applies random mutations to a population of agents — can mutate architecture, hidden layer sizes, activations, learning rates, batch sizes, etc.
- `TournamentSelection` (`hpo/tournament.py`): Selects the fittest agents from a population to seed the next generation

**Training Loops** (`agilerl/training/`)
- High-level functions (e.g., `train_off_policy`, `train_on_policy`) manage the full population-based training loop: collect experience → learn → tournament select → mutate
- These are optional convenience wrappers; users can also drive the loop manually

**Components** (`agilerl/components/`)
- Replay buffers (standard, prioritized, multi-step), rollout buffers, samplers
- These are used directly by algorithms and are not evolvable

### LLM Fine-tuning

LLM algorithms (`grpo.py`, `dpo.py`, `bc_lm.py`, `ppo_llm.py`) live alongside standard RL algorithms but depend on optional packages (transformers, peft, vllm). The `HAS_LLM_DEPENDENCIES` flag in `__init__.py` controls availability. LLM utilities are in `agilerl/utils/llm_utils.py`.

### Multi-Agent

Multi-agent algorithms (`maddpg.py`, `matd3.py`, `ippo.py`) follow the same evolvable pattern but use `MultiAgentRLAlgorithm` as their base. They are compatible with PettingZoo environments.

### Protocols

`agilerl/protocols.py` defines the structural interfaces (`EvolvableAlgorithmProtocol`, `EvolvableModuleProtocol`, etc.) used for type checking. Code that accepts any evolvable algorithm should type against these protocols rather than concrete base classes.

### Configuration

Training configs live in `configs/training/` as YAML files and are loaded via Hydra/OmegaConf. Benchmarking/debugging scripts are in `benchmarking/`.
