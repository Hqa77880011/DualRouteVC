# DualRouteVC

Implementation of **DualRouteVC: Motion-Uncertainty-Aware Scalable Neural Video Coding for Human-Machine Perception**.

A quantized base latent reconstructs each current frame relative to its preceding frame. A bounded local displacement posterior supplies entropy and dominant flow. A second quantized latent encodes residual and motion uncertainty for optional correspondence refinement. The base stream alone remains viewable. Rate is a logistic-mass entropy objective; actual serialized latent streams can be compressed independently with the codec helper.

## Install

Use Python 3.10 or later. Create and activate a virtual environment, then install:

```bash
python -m pip install -e .
```

## Prepare data

`x[N,6,H,W]` concatenates previous and current RGB frames and `y[N,3,H,W]` is the current target. Optional `flow[N,2,H,W]` is reference optical flow for EPE. Use Vimeo-90K/UVG-style disjoint video groups; evaluate downstream frozen models separately on decoded frames and supplied correspondence fields.

Arrays use numeric NPZ format, loaded without pickle. `group[N]` identifies independent subjects, devices, scenes, lots, or source images. Related samples must share a group. Optional `split[N]` codes 0/1/2/3 explicitly assign training/validation/calibration/test (three-role input using 0/1/2 remains supported); otherwise ordered groups are allocated 60/15/10/15 to training/validation/calibration/test. Test data never select model parameters or checkpoints.

```bash
dualroutevc demo --output data/example.npz --count 96 --config config.json
dualroutevc prepare --input data/example.npz --output data/prepared
```

The demo defines a small synthetic problem with the documented tensor interface. Use application data in that schema for the paper's evaluation settings.

## Train and execute

```bash
dualroutevc train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16
dualroutevc evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
dualroutevc plot --input runs --output runs/figures
```

Training uses AdamW and selects the minimum validation objective. `model.pt` stores parameters and configuration; `history.json` records training/validation objectives. Evaluation exports predictions and mechanism tensors to `predictions.npz` and computed scores to `metrics.json`. Plotting exports comparison charts from those actual evaluation files.

`--device cpu` is the default; `--device cuda` selects CUDA. `--lr`, `--epochs`, `--batch-size`, and `--seed` control optimization. Method parameters are in `config.json`; `--set key=value` overrides one value using JSON syntax. Use identical splits and seeds across comparisons.

## Calibration

For interval or classification confidence calibration, use independent groups excluded from fitting and checkpoint selection:

```bash
dualroutevc calibrate --input data/prepared/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
```

Evaluate `calibrated.pt` with the same evaluation command. Classification uses temperature scaling; numeric intervals use a 90% group-maximum split-calibration expansion. Insufficient independent units retain infinite intervals. Pretraining checkpoints can initialize supervised training with `train --init path/to/model.pt`.

## Baselines and ablations

Available variants: `full`, `baseline`, `no_entropy_gate`, `base_only`. `baseline` removes the paper-specific mechanism while retaining the input and prediction task. Other variants disable their named component.

```bash
dualroutevc train --input data/prepared --output runs/baseline --config config.json --variant baseline
dualroutevc evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Metrics and implementation conventions

Classification reports accuracy, balanced accuracy, macro-F1, NLL, confidence ECE, and worst-group F1; binary tasks add AUROC and AUPRC. Unknown labels use -1 and enable unknown AUROC and OSCR. Segmentation reports Dice, IoU, precision, recall, MAE, Brier/ECE, two-pixel boundary F1, symmetric contour distance, and HD95. Reconstruction reports RMSE, MAE, signed bias, unit-range PSNR, and Gaussian-window SSIM. Each method adds its mechanism metrics where applicable.

Parameter counts and forward timing describe the executing process. Use batch size one and a declared device configuration for device latency comparisons. Physical energy and scores from external task evaluators require measured inputs; they are not inferred from parameter counts. All prediction arrays remain available for task-specific evaluation.

The implementation uses the explicit core operations described above with configurable compact networks. Numeric conventions and application-specific information are stated with the method and data schema. The parameter file contains the current defaults.

## Actual scalable streams

```bash
python -m dualroutevc.codec encode --model runs/full/model.pt --input data/video.npz --output runs/video.drvc
python -m dualroutevc.codec decode --model runs/full/model.pt --input runs/video.drvc --output runs/base.npz --base-only
python -m dualroutevc.codec decode --model runs/full/model.pt --input runs/video.drvc --output runs/enhanced.npz
```

The video archive contains `x[T,3,H,W]` unit-range frames. Both encoder and decoder use the previous base reconstruction, so enhancement is optional. Quantized base, motion, and enhancement arrays are independently DEFLATE coded in a versioned stream; the adjacent JSON reports actual total stream bytes including headers. This concrete coder is distinct from the differentiable logistic entropy objective. Decoder output is an NPZ video in the same frame layout.
