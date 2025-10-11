# MLX Tiny Recursive Models

Simplified reimplementation of [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) using [MLX](https://github.com/ml-explore/mlx).

## Usage

1. Setup the environment

   ```bash
   uv sync
   source .venv/bin/activate
   ```
2. Adjust model config in `train.py`

   ```python
   @dataclass
   class ModelConfig:
       in_channels: int
       depth: int
       dim: int
       heads: int
       patch_size: tuple
       n_outputs: int
       pool: str = "cls" # mean or cls
       n: int = 6  # latent steps
       T: int = 3  # deep steps
       halt_max_steps: int = 8  # maximum supervision steps
       halt_exploration_prob: float = 0.2  # exploratory q probability
       halt_follow_q: bool = True  # follow q (True) or max steps (False)
   ```
3. Train on MNIST or CIFAR-10 (see `python train.py --help`):
   ```bash
   python train.py --dataset mnist
   python train.py --dataset cifar10
   ```

## Notes

- Hyperparams are currently hardcoded for faster experimentation.
- Only MNIST and CIFAR-10 are supported at the moment.
- `uv` handles virtual environment creation automatically.
