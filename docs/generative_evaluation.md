# Generative Evaluation

This report compares VAE, beta-VAE, and CVAE generation quality on terrain heightmap tiles. The evaluation uses 600 real validation tiles per class and 300 generated tiles per class. Metrics combine reconstruction quality, terrain geometry descriptors, distributional distances, and sample diversity.

## Metrics

- `RMSE` and `GradMAE` evaluate reconstruction quality.
- `Rough ratio` and `HF ratio` compare generated or reconstructed texture statistics with real data; the ideal value is close to 1.
- `W1 slope`, `W1 rough`, `W1 hf`, and `W1 range` are Wasserstein distances between real and generated feature distributions.
- `Diversity` is the mean pairwise distance between generated samples on downsampled 32 x 32 maps.

## Summary Table

| Model | Mode | RMSE | Grad MAE | Rough ratio | HF ratio | W1 slope | W1 rough | Diversity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VAE | recon | 0.0911 | 0.0201 | 0.16 | 0.39 | - | - | - |
| VAE | gen | - | - | 0.16 | 0.30 | 0.0212 | 0.0270 | 10.9 |
| beta-VAE | recon | 0.0940 | 0.0213 | 0.71 | 0.38 | - | - | - |
| beta-VAE | gen | - | - | 0.90 | 0.29 | 0.0126 | 0.0167 | 18.1 |
| CVAE | recon | 0.0937 | 0.0204 | 0.25 | 0.38 | - | - | - |
| CVAE | gen | - | - | 0.36 | 0.31 | 0.0158 | 0.0219 | 14.1 |

## Distributional Quality

| Model | W1 slope | W1 roughness | W1 HF energy | W1 range |
|---|---:|---:|---:|---:|
| VAE | 0.0212 | 0.0270 | 0.0607 | 0.758 |
| beta-VAE | 0.0126 | 0.0167 | 0.0615 | 0.208 |
| CVAE | 0.0158 | 0.0219 | 0.0604 | 0.435 |

beta-VAE is closest to the real distribution by slope and roughness. CVAE is slightly weaker on unconditional distributional metrics, but provides controllable generation through terrain class and elevation-range conditioning.

## Diversity

| Source | Diversity |
|---|---:|
| Real tiles | 20.7 |
| VAE | 10.9 |
| beta-VAE | 18.1 |
| CVAE | 14.1 |

All generative models produce varied samples, but beta-VAE is closest to the real diversity reference.

## CVAE Class Consistency

| Feature | Flat | Hilly | Mountain | Monotonic |
|---|---:|---:|---:|:---:|
| Slope | 0.0186 | 0.0202 | 0.0211 | yes |
| Roughness | 0.0109 | 0.0114 | 0.0116 | yes |
| Range | 1.49 | 1.59 | 1.64 | yes |

CVAE conditioning is effective: generated terrain becomes progressively steeper and rougher from `flat` to `hilly` to `mountain`.

## Conclusion

beta-VAE is the strongest unconditional generator, with `rough_ratio = 0.90`, the lowest Wasserstein distances for slope and roughness, and diversity close to real terrain. CVAE is the best controllable option: it maintains comparable reconstruction quality while allowing terrain class steering.
