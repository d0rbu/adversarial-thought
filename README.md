# Adversarial Thought

Research on adversarial attacks against activation oracles in language models.

## Overview

This project investigates **activation oracles as an attack surface** for language model interpretability. The core hypothesis is that models can be finetuned to "think adversarially" - hiding internal secrets and loyalties from activation oracle probes while maintaining task performance.

### Research Questions

1. **Robustness of Activation Oracles**: How much data/training is required to fool activation oracles?
2. **Performance-Obfuscation Tradeoff**: What is the tradeoff between "adversarial thought" and model performance?
3. **Transfer to SAE Methods**: How well do adversarial attacks transfer to Sparse Autoencoder interpretability methods?

## Quick Start

```bash
# Install dependencies
uv sync --all-groups

# Install pre-commit hooks
uv run pre-commit install
uv run pre-commit install --hook-type pre-push

# Run the baseline SFT finetuning
uv run python -m exp.sft_finetune

# Run with custom config overrides
uv run python -m exp.sft_finetune experiment.name=my_exp training.batch_size=8

# Run without W&B logging
uv run python -m exp.sft_finetune wandb.enabled=false
```

## Development

```bash
# Run linting
uv run ruff check --fix .
uv run ruff format .

# Run type checking
uv run ty check

# Run tests
uv run pytest

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

## Project Structure

```
adversarial-thought/
├── conf/           # Hydra configuration files
├── core/           # Core utilities (data loading, types)
├── exp/            # Experiment code
├── test/           # Test suite
├── AGENTS.md       # Detailed documentation for AI assistants
└── pyproject.toml  # Project configuration
```

See [AGENTS.md](AGENTS.md) for detailed documentation.

## License

TBD
