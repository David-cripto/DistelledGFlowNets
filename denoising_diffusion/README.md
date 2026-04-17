# Image Denoising Diffusion

This package ports the `toy_denoising_diffusion` workflow to image data. The first supported dataset is MNIST.

It provides two training stages:

1. Train a DDPM UNet denoiser on the image dataset.
2. Train a detailed-balance reward model `f(x, t)` from a pretrained diffusion model.

## Layout

```text
denoising_diffusion/
├── pyproject.toml
├── README.md
└── src/
    ├── cli.py
    ├── reward_cli.py
    ├── experimental_reward_cli.py
    ├── compatibility_cli.py
    ├── data.py
    ├── diffusion/
    ├── experimental_reward/
    ├── reward/
    └── visualization/
```

## Dataset

The supported datasets are `mnist`, `mnist_0`, and `mnist_1`. The filtered variants keep only a single MNIST digit class. Images are resized to `28x28`, converted to tensors, and standardized with the usual MNIST statistics `Normalize((0.1307,), (0.3081,))`, so the inputs seen by the diffusion model are approximately zero mean and unit variance. The terminal reference distribution is a standard normal over these image tensors.

The diffusion schedule supports both `linear` and `cosine` beta schedules. For low-resolution images, `--beta-schedule cosine` is often the better default.

The noise predictor is a UNet-style encoder-decoder with skip connections and time-conditioned residual blocks. It keeps the existing `model(x_t, t)` interface, but unlike the earlier same-resolution CNN it can aggregate multi-scale context before predicting the noise.

## Diffusion Training

From the `denoising_diffusion/` directory:

```bash
uv run denoising-diffusion-train --dataset mnist --device cpu
```

Single-digit example:

```bash
uv run denoising-diffusion-train --dataset mnist_0 --device cpu
```

Cosine schedule example:

```bash
uv run denoising-diffusion-train --dataset mnist --device cpu --beta-schedule cosine
```

Artifacts are written to `outputs/denoising_diffusion/<dataset>/run` by default and include:

- `checkpoint.pt`
- `real_samples.png`
- `generated_samples.png`
- `sample_trajectory.png`
- `diffusion_training_curves.png`

## Reward Training

Train from a saved diffusion checkpoint:

```bash
uv run denoising-diffusion-reward \
  --diffusion-checkpoint outputs/denoising_diffusion/mnist/run/checkpoint.pt \
  --device cpu
```

Or train diffusion and reward in a single command:

```bash
uv run denoising-diffusion-reward --dataset mnist --device cpu
```

Before the detailed-balance stage, the reward model is first pretrained on Gaussian samples from the reference distribution. It sees random timesteps and regresses to the standard normal log-density, which helps initialize the boundary behavior. Disable this stage with `--reward-pretrain-steps 0`.

Artifacts are written to `outputs/denoising_diffusion/<dataset>/detailed_balance_run` by default and include:

- `detailed_balance_checkpoint.pt`
- `real_samples.png`
- `diffusion_samples.png`
- `terminal_score_histogram.png`
- `boundary_alignment_histogram.png`
- `detailed_balance_training_curves.png`

Only the direct `DetailedBalanceMLP` reward parameterization is implemented in this image version. Despite the name, it now uses a CNN feature extractor over the image and a small time-conditioned MLP head for the final scalar score.

## Experimental Reward

The experimental variant lives in `src/experimental_reward/`. It uses the parameterization

```text
F(x, t) = NN(x, t) - 0.5 * w(t) * ||x||^2
```

where `w(t)` is the normalized diffusion timestep and reaches `1` at the terminal diffusion step. With this form, Gaussian boundary supervision no longer tries to fit the full Gaussian log-density directly. Instead, it drives the neural residual `NN(x, t)` to zero on Gaussian samples, so the total score matches the Gaussian reference up to its additive normalizing constant.

Run it with:

```bash
uv run denoising-diffusion-experimental-reward \
  --diffusion-checkpoint outputs/denoising_diffusion/mnist/run/checkpoint.pt \
  --device cpu
```

By default, artifacts are written to `outputs/denoising_diffusion/<dataset>/experimental_reward_run`.

## Compatibility CLI

The compatibility analysis from the toy 2D package is not ported to image data yet. The CLI is kept only as a placeholder.
