# Game Terrain VAE

Generative modeling project for creating game-ready terrain heightmap tiles from DEM-style elevation data. The repository compares VAE, beta-VAE, and Conditional VAE approaches, evaluates generated terrain with geometric metrics, and includes post-processing utilities for more realistic surface detail.

![Generated terrain examples](figures/final/mountains_gameready.png)

## Project Highlights

- Built a PyTorch pipeline for 256 x 256 single-channel heightmap tiles.
- Implemented AE, VAE, beta-VAE, Conditional VAE, detail refiner, and terrain-fusion experiments.
- Added class-conditional generation for `flat`, `hilly`, and `mountain` terrain.
- Evaluated both reconstruction quality and generative quality with terrain-aware metrics: MAE, RMSE, gradient MAE, slope difference, roughness, high-frequency energy, Wasserstein distances, and diversity.
- Exported visual artifacts for game-oriented review: 2D heightmaps, 3D surfaces, Blender-style renders, and post-processed samples.

## Key Results

### Conditional VAE Reconstruction

Best CVAE validation loss: **0.0143**. On the test split, CVAE reached:

| Metric | Value |
|---|---:|
| MAE | 0.0701 |
| RMSE | 0.0950 |
| Gradient MAE | 0.0218 |
| Slope difference | 0.0267 |
| Approx. MAE in meters | 11.10 m |
| Approx. RMSE in meters | 14.78 m |

Per-class reconstruction:

| Terrain | MAE | RMSE | Approx. MAE |
|---|---:|---:|---:|
| Flat | 0.0745 | 0.1013 | 1.80 m |
| Hilly | 0.0690 | 0.0942 | 7.99 m |
| Mountain | 0.0645 | 0.0856 | 32.39 m |

### Generative Evaluation

The generative comparison used 600 real tiles per class and 300 generated tiles per class.

| Model | Mode | RMSE | Grad MAE | Rough ratio | HF ratio | W1 slope | W1 rough | Diversity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VAE | recon | 0.0911 | 0.0201 | 0.16 | 0.39 | - | - | - |
| VAE | gen | - | - | 0.16 | 0.30 | 0.0212 | 0.0270 | 10.9 |
| beta-VAE | recon | 0.0940 | 0.0213 | 0.71 | 0.38 | - | - | - |
| beta-VAE | gen | - | - | 0.90 | 0.29 | 0.0126 | 0.0167 | 18.1 |
| CVAE | recon | 0.0937 | 0.0204 | 0.25 | 0.38 | - | - | - |
| CVAE | gen | - | - | 0.36 | 0.31 | 0.0158 | 0.0219 | 14.1 |

beta-VAE produced the most realistic unconditional samples by roughness and distributional distance. CVAE was the best option for controllable generation: generated `flat -> hilly -> mountain` samples were monotonic by slope, roughness, and elevation range.

## Method

The data pipeline prepares normalized DEM patches:

- tensor format: `[N, 1, 256, 256]`;
- normalization: per-patch min-max to `[-1, 1]`;
- metadata: terrain class, elevation range, slope statistics, original min/max elevation;
- controllable labels: `flat`, `hilly`, `mountain`.

The CVAE conditions the encoder and decoder on terrain class and normalized elevation range. A gradient-aware reconstruction loss helps preserve terrain structure, while post-processing operations such as erosion, thermal smoothing, power transforms, and warping improve game-readiness.

## Repository Structure

```text
.
├── configs/              # Experiment configs
├── figures/final/        # Selected final visualizations
├── outputs/cvae/         # CVAE metrics and sample grid
├── results/              # Evaluation tables and comparison figures
├── src/
│   ├── data/             # Dataset preparation and terrain labeling
│   ├── evaluation/       # Reconstruction and generative metrics
│   ├── models/           # AE, VAE, beta-VAE, CVAE, refiner, terrain fusion
│   ├── postprocess/      # Terrain enhancement operators
│   ├── training/         # Training entrypoints and losses
│   └── visualization/    # Heightmap and 3D rendering utilities
└── requirements.txt
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare a local dataset:

```bash
python -m src.data.prepare_dataset \
  --config configs/cvae.yaml
```

Train CVAE:

```bash
python -m src.training.train_cvae \
  --config configs/cvae.yaml
```

Run evaluation:

```bash
python -m src.evaluation.eval_models \
  --config configs/cvae.yaml
```

Generate examples:

```bash
python -m src.visualization.generate_examples \
  --config configs/cvae.yaml
```

## Data And Artifacts

Raw DEM data and model checkpoints are intentionally not committed. The repository keeps code, configs, selected metrics, and lightweight visual artifacts. Large local files are ignored by `.gitignore`:

- `data/`
- `checkpoints/`
- `runs/`
- `*.pt`
- `*.ckpt`

## Tech Stack

`Python` · `PyTorch` · `NumPy` · `pandas` · `SciPy` · `Matplotlib` · `PyYAML` · `rasterio` · `DEM processing` · `Generative Modeling`
