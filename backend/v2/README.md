# V2 Side Landmarking Pipeline

This package contains the V2 redesign building blocks:

- Remote side inference client with circuit breaker fallback
- Local fallback adapter that converts current side analyzer output to V2 schema
- Jawline contour solver (`jawline_solver.py`) for Ar/Go/Me refinement
- Calibration runtime + CVAT parser + calibrator trainer
- Offline evaluation script with release gates

## Required V2 response blocks

- `side_analysis.landmarks_v2`
- `side_analysis.quality_v2`
- `scores_v2`

## Additional V2 side debug fields

- `side_analysis.landmarks_v2.method_source`
- `side_analysis.landmarks_v2.gonial_debug`
- `side_analysis.quality_v2.jawline_visibility_score`

## Overlay geometry notes

- Blue gonial overlay now prefers `piecewise_regression` geometry fitted from traced
  `contours.jaw_ramus` (chin-to-ear region) and falls back to legacy hybrid/straight
  paths when regression quality checks fail.
- Green profile overlay includes an added nasion-to-forehead segment, with
  `gonial_debug.naso_frontal_angle`, `gonial_debug.naso_frontal_source`, and
  `gonial_debug.naso_frontal_segment` populated when required landmarks are available.

## Calibration workflow

1. Export CVAT annotations.
2. Normalize labels:

```bash
python -m v2.calibration.cvat_parser --input export.json --output normalized_labels.json
```

3. Build training rows (raw predictions + target labels + quality metadata).
4. Train calibrator:

```bash
python -m v2.calibration.train_calibrator --input train_rows.json --output calibration_model.json
```

5. Enable in backend:

```bash
export V2_CALIBRATION_MODEL_PATH=/path/to/calibration_model.json
```

## Holdout gate evaluation

```bash
python -m v2.evaluate_v2 --input holdout_eval_rows.json --output holdout_report.json
```

Gate thresholds are encoded in `evaluate_v2.py`:

- Ar NME <= 3.0%
- Go NME <= 2.5%
- Me NME <= 2.0%
- Gonial MAE <= 3.5 deg
