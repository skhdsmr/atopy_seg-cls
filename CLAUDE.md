# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research workspace for atopic-dermatitis (아토피) images: lesion/skin **segmentation** + **severity
classification** on the EASI axes. Most work happens in many self-contained experiment folders
(`cls_*/`, `classification_*/`, `segmentation/`), not in a single package. Code comments, READMEs and
logs are in Korean — keep that style when editing.

The root `README.md` is partly stale: it describes `classification/`, `app/app.py` (Streamlit) and
`cls_ccnn/`, which are deleted in the working tree. `app/` now holds an unrelated mobile app
(DermaScan: `app/dermascan-app` Expo + `app/server` FastAPI + `app/model`); see `app/README.md`.

Git tracks only `segmentation/`, three old `app/` files, the since-deleted `classification/` and
`cls_ccnn/`, and root README/requirements (single "Initial commit"); everything else, including
all `cls_*` experiment folders, is untracked local work. `.gitignore` forbids committing `atopy/` and
`dataset_*` (patient images, privacy) and all `runs/`.

## Environment (important)

The container's global `PYTHONPATH`/`LD_LIBRARY_PATH` point at a CPU torch build, so plain
`python3` gives broken/CPU torch. All training runs through the CUDA venv `.venv-train` with an
isolated env — every `run.sh` does this preamble; copy it for new scripts or ad-hoc commands:

```bash
export PYTHONPATH=""
PY=/home/work/ogw/.venv-train/bin/python
SP=$($PY -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
export LD_LIBRARY_PATH="$SP/torch/lib:$(printf '%s:' $SP/nvidia/*/lib)/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
```

`.venv-train` has torch 2.2 (cu121), timm 1.0, segmentation_models_pytorch 0.5.0, scipy, sklearn,
cv2 — but **not pandas**; the system `python3` has pandas/sklearn/cv2 but no usable CUDA torch. Use
system python3 for pandas-based analysis, the venv for anything touching models. torch 2.2 means
`torch.cuda.amp.GradScaler` (not `torch.amp.GradScaler`). Hardware: 1× 80 GB GPU, 12 CPU cores —
run ~2 training jobs concurrently at most (dataloader workers are the bottleneck).

There is no test suite, linter or build. "Running" = launching a training/eval script.

## Running experiments

Each experiment folder follows the same pattern: `train.py` (argparse) + `run.sh` / `run_sweep.sh`
that set the env above and take **env-var knobs** documented in the script header, e.g.

```bash
bash cls_atopy_mbn/run.sh                      # defaults
ASPP=0 IMGSZ=448 bash cls_atopy_mbn/run.sh     # knobs are env vars
bash segmentation/encoder_decoder/run.sh       # seg training (smp U-Net++ etc.)
```

Outputs go to `<folder>/runs/<name>/` (`best.pt`, `last.pt`, `test_report.json`, sometimes
`done.txt` used to skip finished runs unless `FORCE=1`). Long jobs are launched detached so they
survive the IDE closing, with logs in `<folder>/logs/` — see `cls_papu/run.sh` for the
`setsid nohup` queue pattern (`TAG`, `SEEDS`, `SCALAR` knobs; `tail -f logs/queue_<TAG>.log`).

`/home/work/Code/` is a separate offline-server copy of some projects (`cls_sev`, `cls_atopy_sev`,
`cls_kd`, …) plus `Code/weights/` (domain-pretrained checkpoints such as
`effunet_pvtv2b0_512.pt`, a 5-axis pvt_v2_b0 whose `backbone.*` keys load strictly into timm
`pvt_v2_b0`). Work in `ogw/` unless told otherwise; `ogw` copies use timm with internet
(`pretrained=True`), `Code/` copies vendor backbones.

## Data conventions

- `dataset_all_final/` (1,800 imgs; train 1400 / val 200 / test 200): `images/{split}/*.png`,
  `labels/{split}/*.txt` = **YOLO polygon** lesion masks (single class 0, normalized coords),
  `images/labels.csv` = `split,stem,severity,erythema,papulation,excoriation,lichenification`.
  `severity` is the IGA axis (5 grades Clear…Severe); the 4 signs are None/Mild/Moderate/Severe.
  The split is defined by folders, not the CSV column.
- Pandas gotcha: read `labels.csv` with `keep_default_na=False`, otherwise the grade `None`
  becomes NaN.
- Images come from two sources distinguishable only by original resolution (512 vs 1024, 900
  each, balanced across splits). Framing differs hugely (lesion area ≈0.63 vs ≈0.20 of frame) and
  grade distributions differ (e.g. papulation Severe is 82% in 1024). Pooled area↔grade
  correlations cancel out (Simpson's paradox) — normalize mask-derived features per source.
- `atopy_crop_masks/{split}/*.png`: 1024² binary lesion masks for the same stems (used by the
  crop-classification experiments).
- `dataset_rmask/` (built by `make_rmask.py`): images skin-segmented with
  `seg_skin_effb0unetpp_512.pt` (smp UnetPlusPlus `tu-efficientnet_b0`, ckpt dict with `model`,
  `args`) → cropped to skin bbox, non-skin black; `labels/{split}/*.png` = 0/255 lesion masks from
  `seg_best(@512).pth` (raw state_dict, smp UnetPlusPlus `efficientnet-b0`). Both models: resize
  512 + ImageNet norm, sigmoid > 0.5. 242 lesion masks are empty, all from the 1024 source.

## Classification architecture (shared across cls_* folders)

Folders are deliberately self-contained: shared logic (`labels.py`/`dataset.py`/`metrics.py`,
backbone builder) is **copied** into each folder rather than imported, so a change in one folder
does not propagate. Common design across them:

- Backbone via `timm.create_model(name, num_classes=0, global_pool="avg")` (feature dim probed
  with a dummy forward) → optional neck → per-axis heads.
- Ordinal **CORN** loss (K-1 logits, conditional binary tasks; predict via cumprod > 0.5) is the
  default; CE is the alternative. Multi-task runs combine the 5 axes with Kendall uncertainty
  weighting (`--mtl uncertainty`).
- Class weights = inverse frequency^0.5, mean-normalized, clipped to 3.
- Primary metric is **QWK**; multi-task model selection = 0.5·IGA QWK + 0.5·mean sign QWK.
- Crop experiments (`classification_crop`, `Code/cls_atopy_sev`): `full` / `bbox` (mask union bbox
  + margin) / `mil` (connected-component bag) / `twostream`. Background is **not** zeroed in these
  (zeroing background was measured to hurt).
- `cls_papu/`: papulation single-task on `dataset_rmask`, square bbox crop + mask scalars (area,
  bbox, fill, log1p #components) z-scored per source with train stats excluding empty masks, plus
  a `no_mask` flag; options `--weights` (domain backbone), `--stop_on {loss,qwk}`, `--ema`.
- `cls_kd/`: BAM born-again multi-task distillation (conditions A–H in its README).

Test sets are small (200 images, ≈±0.05 QWK standard error): compare settings over several seeds,
and report per-source (512/1024) metrics.

## Segmentation

`segmentation/` holds shared modules (`dataset.py`, `augment.py` with ImageNet norm, `losses.py`,
`metrics.py`, `tiling.py`); `encoder_decoder/model.py::build_model` wraps smp decoders
(`unetpp`, `unet`, `manet`) plus a ported `emcad.py`. UNet++ cannot take transformer encoders
without a 1/2-scale stage (pvt/swin/mit) — use `unet`/`emcad` for those. Inference/export helpers
(`infer_all_splits.py`, `make_atopy_seg.py`) share the same preprocessing: resize to ckpt `imgsz`,
ImageNet norm, bilinear upsample logits to original size, sigmoid > 0.5.
