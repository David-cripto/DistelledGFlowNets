# Denoising Diffusion Playground

This package contains a small 2D DDPM implementation plus a detailed-balance experiment for learning a GFlowNet-style score function over diffusion transitions.

## Layout

```text
denoising_diffusion/
├── pyproject.toml
├── README.md
└── src/
    ├── cli.py
    ├── reward_cli.py
    ├── diffusion/
    ├── distributions/
    ├── reward/
    └── visualization/
```

## Diffusion Model

The diffusion code follows the DDPM parameterization from Ho et al. (2020):

`x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps`

where `x_0` comes from the target density and `eps ~ N(0, I)`. The network predicts the injected Gaussian noise, and sampling uses the DDPM reverse posterior.

The terminal prior is fixed to the standard normal, so `--reference` must be `gaussian`.

Available target distributions:

- `gaussian`
- `wide_gaussian`
- `shifted_gaussian`
- `mixture4`
- `grid9`
- `eight_gaussians`
- `ring`
- `two_moons`

## Run

From the `denoising_diffusion/` directory:

```bash
uv run denoising-diffusion-train --target two_moons --reference gaussian
```

Or run the module directly:

```bash
PYTHONPATH=. python -m src.cli --target ring --reference gaussian
```

Artifacts are written to `outputs/denoising_diffusion/run` by default.

## Detailed Balance Score Learning

Given a trained diffusion model, the reward module learns a scalar field

`f(x, t) = log F(x, t)`

with a soft boundary condition that penalizes the mismatch between `f(x, 1)` and the Gaussian prior log-density.

Training minimizes the detailed-balance residual

`(f(x_{t-1}, t-1) + log q(x_t | x_{t-1}) - f(x_t, t) - log p(x_{t-1} | x_t))^2`

over DDPM transition pairs sampled from the reverse model. The forward kernel `q` is the known diffusion Gaussian, and the reverse density `p` uses the DDPM model mean together with the fixed variance `beta_t`. The same reverse kernel is used by the diffusion sampler, so the reward objective is trained against the actual transitions produced by the denoiser. The total loss is the detailed-balance term plus a configurable boundary penalty toward the prior log-density at `t=1`.

The reward CLI supports three parameterizations for `F` through `--reward-model`:

- `direct`: learn `f(x, t)` directly
- `target_factored`: `F(x, t) = \tilde{F}(x, t) R_\theta(x)` with `\tilde{F}(x, 0) = 1`, implemented as `f(x, t) = r_\theta(x) + t * g_\theta(x, t)`
- `denoised_target_factored`: `F(x, t) = \tilde{F}(x, t) R_\theta(\hat{x}(x, t))` with `\tilde{F}(x, 0) = 1`, where `\hat{x}(x, t)` is the DDPM prediction of `x_0`

Run it with:

```bash
uv run denoising-diffusion-reward --target two_moons --reference gaussian --reward-model denoised_target_factored
```

Or:

```bash
PYTHONPATH=. python -m src.reward_cli --target two_moons --reference gaussian --reward-model target_factored
```

Artifacts are written to `outputs/denoising_diffusion/detailed_balance_run` by default. The main outputs are:

- `exp_f_t1.png`: the requested plot of `exp(f(x, 1))`, which should approximate the Gaussian prior density
- `exp_f_t0.png`: the learned score field at the data end of the chain
- `terminal_score_density_comparison.png`: target density, diffusion KDE, and normalized `exp(f(x, 0))`
- `detailed_balance_training_curves.png`

## Kernel Compatibility Check

There is also a separate compatibility tool that tests whether the learned reverse kernel `p` and the known forward kernel `q` are mutually consistent on a grid. It fixes `\pi_T` to the Gaussian prior, propagates backward with `p`, pushes the result forward with `q`, and compares the roundtrip density to the original marginal at every time slice.

Run it with:

```bash
uv run denoising-diffusion-compatibility --checkpoint outputs/denoising_diffusion/run/checkpoint.pt
```

Or train a small model and check compatibility in one command:

```bash
PYTHONPATH=. python -m src.compatibility_cli --target two_moons --reference gaussian
```

Artifacts are written to `outputs/denoising_diffusion/compatibility_run` by default. The main outputs are:

- `compatibility_step_metrics.png`: per-step raw and normalized roundtrip errors plus mass diagnostics
- `compatibility_selected_steps.png`: selected time slices showing `\pi_t`, `q_\# \pi_{t-1}`, and their difference
- `terminal_vs_target.png`: the backward terminal marginal compared against the exact target density
