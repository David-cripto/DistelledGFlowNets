# Image Denoising Diffusion

This package ports the `toy_denoising_diffusion` workflow to image data. The first supported dataset is MNIST.

It provides two training stages:

1. Train a DDPM denoiser on the image dataset.
2. Train a detailed-balance reward model `f(x, t)` from a pretrained diffusion model.

## Layout

```text
denoising_diffusion/
├── pyproject.toml
├── README.md
└── src/
    ├── cli.py
    ├── reward_cli.py
    ├── compatibility_cli.py
    ├── data.py
    ├── diffusion/
    ├── reward/
    └── visualization/
```

## Dataset

Only `mnist` is implemented right now. Images are standardized before training with the dataset statistics `mean=0.1307` and `std=0.3081`, so the diffusion model sees approximately zero-mean, unit-variance inputs. The terminal reference distribution is a standard normal over these standardized image tensors.

## Diffusion Training

From the `denoising_diffusion/` directory:

```bash
uv run denoising-diffusion-train --dataset mnist --device cpu
```

Artifacts are written to `outputs/denoising_diffusion/run` by default and include:

- `checkpoint.pt`
- `real_samples.png`
- `generated_samples.png`
- `sample_trajectory.png`
- `diffusion_training_curves.png`

## Reward Training

Train from a saved diffusion checkpoint:

```bash
uv run denoising-diffusion-reward \
  --diffusion-checkpoint outputs/denoising_diffusion/run/checkpoint.pt \
  --device cpu
```

Or train diffusion and reward in a single command:

```bash
uv run denoising-diffusion-reward --dataset mnist --device cpu
```

Artifacts are written to `outputs/denoising_diffusion/detailed_balance_run` by default and include:

- `detailed_balance_checkpoint.pt`
- `real_samples.png`
- `diffusion_samples.png`
- `terminal_score_histogram.png`
- `boundary_alignment_histogram.png`
- `detailed_balance_training_curves.png`

Only the direct `DetailedBalanceMLP` reward parameterization is implemented in this image version. Despite the name, it now uses a CNN feature extractor over the image and a small time-conditioned MLP head for the final scalar score.

## Compatibility CLI

The compatibility analysis from the toy 2D package is not ported to image data yet. The CLI is kept only as a placeholder.
