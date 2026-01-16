# OpenPI

OpenPI is an open-source repository by the Physical Intelligence team, containing models and packages for robotics. It features the $\pi_0$ (Pi-Zero) family of Vision-Language-Action (VLA) models, including $\pi_0$, $\pi_0$-FAST, and $\pi_{0.5}$.

## Project Overview

- **Core Models:**
    - **$\pi_0$ (Pi-Zero):** A flow-based VLA model.
    - **$\pi_0$-FAST:** An autoregressive VLA model using FAST action tokenization.
    - **$\pi_{0.5}$:** An upgraded version of $\pi_0$ with improved open-world generalization.
- **Frameworks:**
    - Primary implementation in **JAX** (using Flax and Equinox).
    - **PyTorch** support recently added (beta) for $\pi_0$ and $\pi_{0.5}$.
- **Package Management:** Uses `uv` for fast Python dependency management.
- **Data:** Uses LeRobot datasets.

## Key Directories

- **`src/openpi/`**: Main library source code.
    - **`models/`**: JAX model definitions (Pi0, Gemma, SigLIP).
    - **`models_pytorch/`**: PyTorch implementations of the models.
    - **`policies/`**: Policy abstractions and specific implementations (Aloha, Droid, Libero).
    - **`training/`**: Training configuration, data loaders, and loops.
    - **`serving/`**: Websocket-based policy server.
- **`scripts/`**: Entry points for core tasks.
    - `train.py`: JAX training script.
    - `train_pytorch.py`: PyTorch training script.
    - `serve_policy.py`: Policy serving script.
    - `compute_norm_stats.py`: Utility to compute dataset statistics.
- **`examples/`**: Integration examples for various robot platforms (Aloha, Droid, Libero, UR5) and data conversion scripts.

## Building and Running

### Installation
The project uses `uv`.

```bash
uv sync
```

### Training (JAX)
To train a model (e.g., Pi0.5 on Libero):

```bash
# 1. Compute normalization statistics
uv run scripts/compute_norm_stats.py --config-name pi05_libero

# 2. Run training
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 is recommended for JAX
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment
```

### Training (PyTorch)
PyTorch training follows a similar pattern but uses a different script:

```bash
uv run scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_test
```

### Serving
To serve a trained policy (JAX or PyTorch):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=path/to/checkpoint
```

## Development Conventions

- **Dependency Management:** strictly use `uv`.
- **Linting & Formatting:** `ruff` is configured in `pyproject.toml`.
- **Type Checking:** Codebase is typed (`py.typed` present).
- **Configuration:** Uses `tyro` and `ml_collections` (via wrapper) for configuration.
- **Testing:** `pytest` is used. Run tests via `uv run pytest`.
