# AGENTS.md - Adversarial Thought Research Project

## Project Overview

This project investigates **activation oracles as an attack surface** for language model interpretability. The core hypothesis is that models can be finetuned to "think adversarially" - hiding internal secrets and loyalties from activation oracle probes while maintaining task performance.

### Research Questions (Priority Order)

1. **Robustness of Activation Oracles**: How much data/training is required to fool activation oracles when actors intentionally obfuscate their model's secrets?
2. **Performance-Obfuscation Tradeoff**: What is the tradeoff between "adversarial thought" and model performance? How does this scale?
3. **Transfer to SAE Methods**: How well do adversarial attacks on activation oracles transfer to Sparse Autoencoder (SAE) interpretability methods?
4. **Reverse Transfer**: Do attacks on SAE methods transfer back to activation oracles?

## Repository Structure

```
adversarial-thought/
├── conf/                   # Hydra configuration files
│   ├── config.yaml         # Main config for training
│   ├── eval_config.yaml    # Main config for evaluation
│   ├── oracle_config.yaml  # Main config for oracle evaluation
│   ├── model/              # Model configurations
│   │   ├── gemma3_1b.yaml  # Gemma3-1B config
│   │   └── qwen3_8b.yaml   # Qwen3-8B config
│   ├── data/               # Dataset configurations
│   │   └── dolci_sft.yaml  # Dolci-Instruct-SFT config
│   ├── training/           # Training configurations
│   │   └── default.yaml    # Default training hyperparameters
│   ├── eval/               # Evaluation configurations
│   │   ├── baseline.yaml   # Baseline model eval tasks
│   │   ├── quick.yaml      # Quick eval for testing
│   │   └── sft.yaml        # SFT finetuned model eval tasks
│   └── oracle/             # Oracle evaluation configurations
│       ├── default.yaml    # Default oracle eval config
│       ├── quick.yaml      # Quick oracle eval for testing
│       ├── quick_dataset.yaml # Quick oracle eval with dataset
│       └── sft.yaml        # Oracle eval for SFT finetuned models
├── core/                    # Core utilities
│   ├── data.py             # Dataset loading utilities
│   ├── dtype.py            # Torch dtype utilities
│   ├── questions.py        # Activation oracle question sets
│   └── type.py             # Type assertion utilities
├── exp/                     # Experiment code
│   ├── sft_finetune.py     # SFT baseline finetuning
│   ├── evaluate.py         # Model evaluation with lm-eval
│   ├── oracle.py           # Activation oracle utilities
│   └── run_oracle.py       # Oracle evaluation script
├── test/                    # Test suite
│   ├── conftest.py         # Pytest fixtures
│   ├── test_core.py        # Core utility tests
│   ├── test_data.py        # Data loading tests
│   ├── test_evaluate.py   # Evaluation tests
│   ├── test_exp.py         # Experiment tests
│   ├── test_oracle.py      # Oracle tests
│   ├── test_questions.py  # Questions module tests
│   └── test_sft_finetune.py # SFT finetuning tests
├── script/                  # Shell scripts for running experiments
│   ├── run_sft.sh          # Run SFT finetuning
│   ├── run_eval.sh         # Run evaluation
│   ├── run_eval_baseline.sh # Run baseline evaluation
│   ├── run_eval_sft.sh     # Run SFT evaluation
│   ├── run_eval_sft_quick.sh # Quick SFT evaluation
│   ├── run_oracle.sh       # Run oracle evaluation
│   ├── run_oracle_baseline.sh # Run baseline oracle eval
│   ├── run_oracle_sft.sh   # Run SFT oracle eval
│   ├── run_oracle_quick.sh # Quick oracle eval
│   ├── run_oracle_quick_gemma.sh # Quick oracle eval with Gemma
│   └── run_oracle_sft_quick.sh # Quick SFT oracle eval
├── out/                     # Experiment outputs (model checkpoints, results)
├── hydra/                   # Hydra output logs
├── .pre-commit-config.yaml  # Pre-commit hooks
├── pyproject.toml          # Project configuration
├── README.md               # Project README
└── AGENTS.md               # This file
```

## Setup & Development

### Initial Setup

```bash
# Install dependencies with uv
uv sync --all-groups

# Install pre-commit hooks
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

### Development Commands

```bash
# Run linting with auto-fix
uv run ruff check --fix .
uv run ruff format .

# Run type checking
uv run ty check

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=core --cov=exp --cov-report=term-missing

# Run all pre-commit hooks manually
uv run pre-commit run --all-files
```

### Running Experiments

```bash
# Run baseline SFT finetuning with default config
uv run python -m exp.sft_finetune

# Override config values via CLI
uv run python -m exp.sft_finetune experiment.name=my_exp training.batch_size=8

# Use different config combinations
uv run python -m exp.sft_finetune model=qwen3_8b data=dolci_sft

# Disable W&B logging for local runs
uv run python -m exp.sft_finetune wandb.enabled=false
```

### Running Evaluation

```bash
# Evaluate base model with default config
uv run python -m exp.evaluate

# Quick evaluation (limited samples for testing)
uv run python -m exp.evaluate eval=quick

# Baseline model evaluation
uv run python -m exp.evaluate eval=baseline

# SFT finetuned model evaluation
uv run python -m exp.evaluate eval=sft

# Evaluate a finetuned model with PEFT adapter
uv run python -m exp.evaluate eval.peft_adapter_path=out/sft_baseline

# Evaluate specific tasks
uv run python -m exp.evaluate eval.tasks='[hellaswag,winogrande]'

# Limit examples per task (for quick testing)
uv run python -m exp.evaluate eval.limit=100 eval.tasks='[hellaswag]'

# Disable W&B logging
uv run python -m exp.evaluate wandb.enabled=false
```

### Running Oracle Evaluation

```bash
# Run oracle evaluation with default config
uv run python -m exp.run_oracle

# Quick oracle evaluation (limited questions for testing)
uv run python -m exp.run_oracle oracle=quick

# Oracle evaluation on SFT finetuned model
uv run python -m exp.run_oracle oracle=sft

# Oracle evaluation with dataset
uv run python -m exp.run_oracle oracle=quick_dataset

# Override number of questions
uv run python -m exp.run_oracle questions.n_questions=10

# Use different question split (train, val, or all)
uv run python -m exp.run_oracle questions.split=val

# Evaluate specific model adapter
uv run python -m exp.run_oracle oracle.target_adapter_path=out/sft_baseline

# Disable W&B logging
uv run python -m exp.run_oracle wandb.enabled=false
```

**Note**: Oracle evaluation requires `OPENAI_API_KEY` environment variable for the LLM judge. Set it in your environment or in a `.env` file in the project root.

## Configuration System

We use [Hydra](https://hydra.cc/) for configuration management. The main configs are:

- **config.yaml**: Main config for SFT training experiments
- **eval_config.yaml**: Main config for model evaluation
- **oracle_config.yaml**: Main config for activation oracle evaluation

Config groups:
- **model/**: Model architecture, tokenizer, and LoRA settings
- **data/**: Dataset loading, splits, and tokenization
- **training/**: Training hyperparameters and optimization
- **eval/**: Evaluation tasks and settings (baseline, quick, sft)
- **oracle/**: Oracle evaluation settings (default, quick, quick_dataset, sft)

### Config Overrides

Hydra supports hierarchical config overrides:

```bash
# Change learning rate
uv run python -m exp.sft_finetune training.learning_rate=1e-5

# Change model
uv run python -m exp.sft_finetune model.name=google/gemma-3-4b-it

# Change multiple values
uv run python -m exp.sft_finetune \
    training.num_epochs=5 \
    experiment.seed=123
```

## Experiment Roadmap

### Phase 1: Baseline (Current)
- [x] Set up SFT finetuning on Dolci-Instruct-SFT
- [x] Add lm-eval-harness integration for benchmarking
- [ ] Establish baseline metrics on lm-eval-harness benchmarks
- [x] Evaluate activation oracle accuracy on baseline model

### Phase 2: Oracle Question Generation
- [x] Design question set Q for activation oracle probing
- [x] Generate QA pairs from oracle on SFT data
- [x] Implement LLM judge for answer quality assessment
- [x] Implement oracle evaluation infrastructure (`exp/oracle.py`, `exp/run_oracle.py`)
- [x] Add oracle evaluation configurations

### Phase 3: Adversarial Finetuning
- [x] Implement adversarial loss term (maximize oracle error)
- [x] Train adversarial models with varying loss weights
- [ ] Analyze performance-obfuscation Pareto frontier

### Phase 4: Transfer & Analysis
- [ ] Test adversarial models against SAE probes
- [ ] Analyze internal representations
- [ ] Document findings and write up results

## Code Style & Quality

### Linting & Formatting
- **Ruff** for linting and formatting (replaces black, isort, flake8)
- Line length: 88 characters
- Python 3.12+ features encouraged

### Type Checking
- **ty** for type checking (`uv run ty check`)
- Use type hints for all function signatures
- Use `from __future__ import annotations` for modern syntax

### Testing
- **pytest** for testing
- Tests in `test/` directory
- Aim for >80% coverage on core modules

### Pre-commit Hooks
The following hooks run automatically:
1. **ruff check --fix**: Lint and auto-fix issues
2. **ruff format**: Format code
3. **ty check**: Type check (on commit)
4. **pytest**: Run tests (on push only)

## Agent Guidelines

When working on this codebase:

1. **Follow existing patterns**: Use existing utilities in `core/` before adding new ones
2. **Use Hydra configs**: All hyperparameters should be configurable via Hydra
3. **Write tests**: Add tests for new functionality in `test/`
4. **Type everything**: Use type hints consistently
5. **Document changes**: Update this AGENTS.md when adding new modules or changing structure
6. **Run quality checks**: Always run `uv run ruff check --fix` and `uv run ty check` before committing

### Adding New Experiments

1. Create new config files in `conf/` if needed
2. Add experiment module in `exp/`
3. Use `@hydra.main()` decorator with config path
4. Add corresponding tests in `test/`
5. Optionally add shell script in `script/` for convenience

### Adding New Datasets

1. Add loading function in `core/data.py`
2. Register in `DATASETS` dictionary
3. Create config in `conf/data/`
4. Add tests for the new loader

## Questions Module

The `core/questions.py` module contains questions for probing activation oracles:

- `TRAIN_QUESTIONS`: Questions for generating oracle answers during adversarial training
- `VAL_QUESTIONS`: Held-out questions for evaluation

### Usage

```python
from core.questions import TRAIN_QUESTIONS, VAL_QUESTIONS, to_chat_message

# Access questions directly
for q in TRAIN_QUESTIONS:
    print(q)

# Convert to chat format for tokenizer
messages = to_chat_message("What is the topic?")
# Returns: [{"role": "user", "content": "What is the topic?"}]
```

## Oracle Module

The `exp/oracle.py` module provides utilities for activation oracle evaluation:

- `OracleConfig`: Configuration dataclass for oracle evaluation
- `run_oracle_eval()`: Main function to run oracle evaluation on context-question pairs
- Integration with `nl_probes` library for activation probing
- LLM judge scoring (requires `OPENAI_API_KEY`)

### Usage

```python
from exp.oracle import OracleConfig, run_oracle_eval

config = OracleConfig(
    model_name="Qwen/Qwen3-8B",
    oracle_path="adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
    target_adapter_path="out/sft_baseline",
)
context_question_pairs = [
    ("The capital of France is Paris.", "What is the topic?"),
]
results = run_oracle_eval(config, context_question_pairs)
```

The `exp/run_oracle.py` script provides a Hydra-based CLI for running oracle evaluations with various configurations.

## Dependencies

Key dependencies:
- **transformers**: HuggingFace transformers for model loading
- **peft**: Parameter-Efficient Fine-Tuning (LoRA)
- **datasets**: HuggingFace datasets
- **lm-eval**: Language model evaluation harness
- **hydra-core**: Configuration management
- **wandb**: Experiment tracking
- **accelerate**: Distributed training support
- **torch**: PyTorch backend
- **loguru**: Logging library
