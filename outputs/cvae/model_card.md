# Model Card: Conditional VAE

Conditional beta-VAE for controllable heightmap generation by terrain type (`flat`, `hilly`, `mountain`) and continuous elevation range.

## Architecture

- Base model: convolutional CVAE with GroupNorm and SiLU activations.
- Input: single-channel 256 x 256 heightmap tile.
- Latent dimension: 128.
- Conditioning: terrain class embedding plus normalized elevation-range projection.
- Approximate parameters: 18.99M.

## Objective

The model uses the beta-VAE objective with an additional gradient-aware reconstruction term:

```text
MSE reconstruction + beta * KL / num_pixels + gradient loss
```

Training setup:

- beta: 0.5;
- KL annealing: 6 epochs;
- gradient loss weight: 0.05;
- trained epochs: 12;
- best epoch: 11.

## Data

- Tensor format: `[N, 1, 256, 256]`.
- Normalization: per-patch min-max to `[-1, 1]`.
- Scale conditioning: `log1p(elevation_range)` standardized by train statistics.
- Scale normalization stats: mean `5.0540`, std `1.3171`.

## Test Metrics

| Metric | Value |
|---|---:|
| Best validation loss | 0.0143 |
| MAE | 0.0701 |
| RMSE | 0.0950 |
| Gradient MAE | 0.0218 |
| Slope difference | 0.0267 |
| Roughness difference | 0.0285 |
| Approx. MAE in meters | 11.10 m |
| Approx. RMSE in meters | 14.78 m |

## Controllability

Generated samples follow the expected terrain-class ordering: median slope, roughness, and elevation range increase from `flat` to `hilly` to `mountain`. This makes the model suitable for controllable terrain prototyping rather than only unconditional sampling.

## Limitations

- VAE-family models smooth high-frequency terrain texture.
- Per-patch min-max normalization limits absolute elevation calibration.
- Additional diffusion or autoregressive refinement would likely improve microrelief realism.
