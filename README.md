# DualRouteVC

Implementation of **DualRouteVC: Motion-Uncertainty-Aware Scalable Neural Video Coding for Human-Machine Perception**.

## Installation

Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.1+cu118, SciPy 1.15.3 and scikit-learn 1.6.1. Checks cover Python syntax, JSON/TOML parsing and CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Generate paired synthetic RGB frames related by a one-pixel shift.

```bash
dualroutevc demo --output data/generated.npz --count 96 --config config.json --seed 13
dualroutevc prepare --input data/generated.npz --output data/prepared
```

`x[N,6,H,W]` concatenates previous and current RGB frames and `y[N,3,H,W]` is the current target. Optional `flow[N,2,H,W]` is reference optical flow for EPE. Use Vimeo-90K/UVG-style disjoint video groups; evaluate downstream frozen models separately on decoded frames and supplied correspondence fields.

Numeric NPZ input uses `group` to keep related samples together. `prepare` writes train/validation/calibration/test archives from ordered groups (60/15/10/15), or honors explicit `split` codes 0/1/2/3.

## Training and inference

```bash
dualroutevc train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16 --seed 13
```

Use `--device cuda` to select CUDA; CPU is the default. Keep the configuration, splits and seed fixed across comparisons.

Generate a video and encode/decode actual scalable streams:

```bash
python -m dualroutevc.codec demo --output data/video.npz --frames 16 --seed 13
python -m dualroutevc.codec encode --model runs/full/model.pt --input data/video.npz --output runs/video.drvc
python -m dualroutevc.codec decode --model runs/full/model.pt --input runs/video.drvc --output runs/base.npz --base-only
python -m dualroutevc.codec decode --model runs/full/model.pt --input runs/video.drvc --output runs/enhanced.npz
```

## Evaluation

```bash
dualroutevc evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
```

## Ablations

Variants: `full`, `baseline`, `no_entropy_gate`, `base_only`. Reuse the prepared data and training seed:

```bash
dualroutevc train --input data/prepared --output runs/baseline --config config.json --variant baseline --epochs 100 --batch-size 16 --seed 13
dualroutevc evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Outputs

- `data/prepared/*.npz`: grouped data splits.
- `runs/full/model.pt`: validation-selected parameters and model configuration.
- `runs/full/history.json` and `training.json`: objectives and run settings.
- `runs/full/evaluation/predictions.npz` and `metrics.json`: predictions/mechanism arrays and computed metrics.

```bash
dualroutevc plot --input runs --output runs/figures
```

`runs/figures/*.png` compares saved evaluation metrics. All data, checkpoint and output paths are CLI arguments.
`runs/video.drvc` stores the coded stream, `video.drvc.json` reports actual bytes/bpp, and the decoded NPZ files contain frames, motion and transported features.
