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
├── conf/                    # Hydra configuration files
│   ├── config.yaml         # Main config with defaults
│   ├── model/              # Model configurations
│   │   └── gemma3_1b.yaml  # Gemma-3-1B-IT config
│   ├── data/               # Dataset configurations
│   │   └── dolci_sft.yaml  # Dolci-Instruct-SFT config
│   └── training/           # Training configurations
│       └── default.yaml    # Default training hyperparameters
├── core/                    # Core utilities
│   ├── data.py             # Dataset loading utilities
│   ├── dtype.py            # Torch dtype utilities
│   └── type.py             # Type assertion utilities
├── exp/                     # Experiment code
│   ├── __init__.py         # Module constants
│   └── sft_finetune.py     # SFT baseline finetuning
├── test/                    # Test suite
│   ├── conftest.py         # Pytest fixtures
│   ├── test_core.py        # Core utility tests
│   └── test_exp.py         # Experiment tests
├── .pre-commit-config.yaml  # Pre-commit hooks
├── pyproject.toml          # Project configuration
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
uv run python -m exp.sft_finetune model=gemma3_1b data=dolci_sft

# Disable W&B logging for local runs
uv run python -m exp.sft_finetune wandb.enabled=false
```

## Configuration System

We use [Hydra](https://hydra.cc/) for configuration management. The main config is composed from:

- **model/**: Model architecture, tokenizer, and LoRA settings
- **data/**: Dataset loading, splits, and tokenization
- **training/**: Training hyperparameters and optimization

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
- [ ] Establish baseline metrics on lm-eval-harness benchmarks
- [ ] Evaluate activation oracle accuracy on baseline model

### Phase 2: Oracle Question Generation
- [ ] Design question set Q for activation oracle probing
- [ ] Generate QA pairs from oracle on SFT data
- [ ] Implement LLM judge for answer quality assessment

### Phase 3: Adversarial Finetuning
- [ ] Implement adversarial loss term (maximize oracle error)
- [ ] Train adversarial models with varying loss weights
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

### Adding New Datasets

1. Add loading function in `core/data.py`
2. Register in `DATASETS` dictionary
3. Create config in `conf/data/`
4. Add tests for the new loader

## Dependencies

Key dependencies:
- **transformers**: HuggingFace transformers for model loading
- **peft**: Parameter-Efficient Fine-Tuning (LoRA)
- **datasets**: HuggingFace datasets
- **hydra-core**: Configuration management
- **wandb**: Experiment tracking
- **accelerate**: Distributed training support
- **torch**: PyTorch backend
- **loguru**: Logging library
