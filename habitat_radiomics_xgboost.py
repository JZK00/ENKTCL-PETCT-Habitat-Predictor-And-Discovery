#!/usr/bin/env python3
"""Reference pipeline: pre-segmented PET/CT habitats -> PyRadiomics -> XGBoost.

This is an illustrative implementation, NOT an exact reproduction of the
published study.

Expected input: aligned 3D PET SUV, CT HU, and integer habitat label map
(0=background, 1/2/3=habitat) for each patient. NIfTI (.nii/.nii.gz) or
other SimpleITK-readable volumetric formats work.

Examples:
  python habitat_radiomics_xgboost.py extract --manifest cases.csv --out radiomics.csv
  python habitat_radiomics_xgboost.py train --train-csv train.csv --valid-csv external.csv --outdir xgb_results
  python habitat_radiomics_xgboost.py predict --csv new_radiomics.csv --model xgb_results/model.joblib --out predictions.csv
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

LOG = logging.getLogger("habitat_radiomics")
HABITATS = (1, 2, 3)
MODALITIES = ("PET", "CT")
FEATURE_CLASSES = ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
METADATA = {"patient_id", "pet_path", "ct_path", "habitat_path", "label", "cohort"}


def _extractor(bin_width, include_shape):
    # Imported here so train/predict work without PyRadiomics installed.
    from radiomics import featureextractor

    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=float(bin_width),
        normalize=False,
        additionalInfo=False,
        minimumROIDimensions=2,
        correctMask=False,
        geometryTolerance=1e-4,
    )
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.disableAllFeatures()
    for name in FEATURE_CLASSES:
        extractor.enableFeatureClassByName(name)
    if include_shape:
        # Shape is determined by the mask and would duplicate between PET & CT.
        extractor.enableFeatureClassByName("shape")
    return extractor


def _aligned(reference, image, name):
    if reference.GetDimension() != 3 or image.GetDimension() != 3:
        raise ValueError(f"{name}: expected 3D image and habitat mask")
    if reference.GetSize() != image.GetSize():
        raise ValueError(f"{name}: voxel grid size differs; register/resample before extraction")
    for field in ("GetSpacing", "GetOrigin", "GetDirection"):
        if not np.allclose(getattr(reference, field)(), getattr(image, field)(),
                           rtol=0, atol=1e-4):
            raise ValueError(f"{name}: {field[3:].lower()} mismatch; align images first")


def _path_from_manifest(manifest_path, raw):
    if pd.isna(raw) or not str(raw).strip():
        raise ValueError("Missing image or mask path")
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = Path(manifest_path).resolve().parent / p
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def extract_table(manifest_path, out_path, pet_bin_width=0.5, ct_bin_width=25.0,
                  min_voxels=10):
    import SimpleITK as sitk

    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    required = {"patient_id", "pet_path", "ct_path", "habitat_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if manifest.empty or manifest.patient_id.isna().any() or manifest.patient_id.duplicated().any():
        raise ValueError("Manifest must have one nonempty row per unique patient_id")
    if min_voxels < 2 or pet_bin_width <= 0 or ct_bin_width <= 0:
        raise ValueError("min_voxels must be >=2 and bin widths must be positive")

    extractors = {
        "PET": _extractor(pet_bin_width, include_shape=True),
        "CT": _extractor(ct_bin_width, include_shape=False),
    }
    records = []
    for i, row in manifest.iterrows():
        patient = str(row.patient_id)
        pet = sitk.ReadImage(str(_path_from_manifest(manifest_path, row.pet_path)))
        ct = sitk.ReadImage(str(_path_from_manifest(manifest_path, row.ct_path)))
        mask = sitk.ReadImage(str(_path_from_manifest(manifest_path, row.habitat_path)))
        _aligned(mask, pet, f"{patient}: PET")
        _aligned(mask, ct, f"{patient}: CT")
        mask_np = sitk.GetArrayFromImage(mask)
        if not np.all(np.isin(np.unique(mask_np), [0, 1, 2, 3])):
            raise ValueError(f"{patient}: habitat label map must contain only 0,1,2,3")
        images = {"PET": pet, "CT": ct}
        item = {"patient_id": patient}
        for col in ("label", "cohort"):
            if col in manifest:
                item[col] = row[col]

        for habitat in HABITATS:
            n = int(np.count_nonzero(mask_np == habitat))
            item[f"H{habitat}_voxel_count"] = n  # Metadata: excluded from XGB by design.
            if n < min_voxels:
                LOG.warning("%s: H%d only %d voxels; skipping radiomics", patient, habitat, n)
                continue
            for modality in MODALITIES:
                try:
                    result = extractors[modality].execute(images[modality], mask, label=habitat)
                except Exception as exc:
                    # Failing for a present ROI is not silently treated as a missing habitat.
                    raise RuntimeError(f"Patient {patient}, {modality}, H{habitat}: {exc}") from exc
                for key, value in result.items():
                    if not key.startswith("original_"):
                        continue  # No diagnostics/provenance/non-radiomics fields.
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    item[f"{modality}_H{habitat}_{key}"] = (
                        number if np.isfinite(number) else np.nan
                    )
        records.append(item)
        LOG.info("Extracted %d/%d: %s", i + 1, len(manifest), patient)

    features = pd.DataFrame.from_records(records)
    if not any(c.startswith(("PET_H", "CT_H")) for c in features.columns):
        raise ValueError("No radiomic features extracted. Inspect habitats and geometry.")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)
    LOG.info("Saved %s (%d patients, %d radiomic features)", out, len(features),
             len([c for c in features if c.startswith(("PET_H", "CT_H"))]))


def _feature_cols(frame):
    # Only radiomics: exclude outcomes, patient IDs, counts, and cohort labels.
    cols = [c for c in frame.columns if c.startswith(("PET_H", "CT_H"))
            and "_original_" in c]
    if not cols:
        raise ValueError("No PET_H*/CT_H* original radiomics columns found")
    if len(cols) != len(set(cols)):
        raise ValueError("Duplicate feature columns")
    return sorted(cols)


def _input_features(frame, feature_names):
    # pandas will fill missing columns with NaN; imputer was learned on training only.
    matrix = frame.reindex(columns=feature_names).apply(pd.to_numeric, errors="raise")
    return matrix.replace([np.inf, -np.inf], np.nan)


def _binary_labels(frame, label_col):
    if label_col not in frame:
        raise ValueError(f"Missing binary outcome column: {label_col}")
    y = pd.to_numeric(frame[label_col], errors="raise").to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{label_col} must contain only 0/1 (no missing outcomes)")
    return y.astype(int)


def _validate_patients(frame, source):
    if "patient_id" not in frame or frame.patient_id.isna().any() or frame.patient_id.duplicated().any():
        raise ValueError(f"{source}: missing/duplicate patient_id")
    if frame.empty:
        raise ValueError(f"{source}: empty dataset")


def _predictions(frame, probabilities, label_col, threshold):
    result = pd.DataFrame({
        "patient_id": frame.patient_id.astype(str).to_numpy(),
        "pred_probability": probabilities,
        "pred_label": (probabilities >= threshold).astype(int),
    })
    if label_col in frame.columns:
        result[label_col] = frame[label_col].to_numpy()
    if "cohort" in frame.columns:
        result["cohort"] = frame.cohort.to_numpy()
    return result


def _metrics(y, probabilities, threshold):
    scores = {"n": int(len(y)), "positive_n": int(y.sum()),
              "accuracy_at_threshold": float(accuracy_score(y, probabilities >= threshold))}
    scores["auc"] = float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None
    return scores


def train_model(train_csv, valid_csv, outdir, label_col="label", k=30,
                threshold=0.5, random_state=42):
    train = pd.read_csv(train_csv, dtype={"patient_id": str})
    _validate_patients(train, train_csv)
    y_train = _binary_labels(train, label_col)
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training data need both outcome classes (0 and 1)")
    valid = None
    if valid_csv:
        valid = pd.read_csv(valid_csv, dtype={"patient_id": str})
        _validate_patients(valid, valid_csv)
        overlap = set(train.patient_id) & set(valid.patient_id)
        if overlap:
            raise ValueError(f"Training/validation patient leakage: {sorted(overlap)[:5]}")
        y_valid = _binary_labels(valid, label_col)
    if not 0 < threshold < 1 or k < 1:
        raise ValueError("threshold must be between 0 and 1; k must be >= 1")

    names = _feature_cols(train)
    x_train = _input_features(train, names)
    # Entirely-missing and zero-variance columns are removed using TRAINING only.
    keep = x_train.notna().any(axis=0) & (x_train.var(axis=0, ddof=0).fillna(0) > 0)
    names = list(x_train.columns[keep])
    if not names:
        raise ValueError("No usable training radiomic features")
    n_select = min(k, len(names))
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(score_func=f_classif, k=n_select)),
        ("xgb", XGBClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=2, reg_lambda=5,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=random_state, n_jobs=4,
        )),
    ])
    pipeline.fit(_input_features(train, names), y_train)
    selected = list(np.asarray(names)[pipeline.named_steps["selector"].get_support()])
    package = {"pipeline": pipeline, "feature_names": names,
               "selected_features": selected, "label_col": label_col,
               "threshold": threshold,
               "note": "Illustrative model; use manuscript-matched preprocessing/selection/hyperparameters."}

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, output / "model.joblib")
    importance = pd.DataFrame({"feature": selected,
                               "importance": pipeline.named_steps["xgb"].feature_importances_})
    importance.sort_values("importance", ascending=False).to_csv(
        output / "feature_importance.csv", index=False)
    reports = {}
    for label, table, y in (("train", train, y_train), ("validation", valid,
                              y_valid if valid is not None else None)):
        if table is None:
            continue
        probs = pipeline.predict_proba(_input_features(table, names))[:, 1]
        _predictions(table, probs, label_col, threshold).to_csv(
            output / f"{label}_predictions.csv", index=False)
        reports[label] = _metrics(y, probs, threshold)
    (output / "metrics.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    LOG.info("Model and results saved in %s; selected %d/%d features", output,
             len(selected), len(names))
    LOG.info("Metrics: %s", json.dumps(reports))


def predict_model(csv_path, model_path, out_path):
    # joblib uses pickle internally. Load only model files from trusted sources.
    package = joblib.load(model_path)
    frame = pd.read_csv(csv_path, dtype={"patient_id": str})
    _validate_patients(frame, csv_path)
    probabilities = package["pipeline"].predict_proba(
        _input_features(frame, package["feature_names"]))[:, 1]
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _predictions(frame, probabilities, package["label_col"], package["threshold"]).to_csv(
        output, index=False)
    LOG.info("Saved %d predictions to %s", len(frame), output)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)
    ext = subs.add_parser("extract", help="Extract PET/CT radiomics from habitat label maps")
    ext.add_argument("--manifest", required=True, help="CSV: patient_id,pet_path,ct_path,habitat_path[,label]")
    ext.add_argument("--out", required=True)
    ext.add_argument("--pet-bin-width", type=float, default=0.5)
    ext.add_argument("--ct-bin-width", type=float, default=25.0)
    ext.add_argument("--min-voxels", type=int, default=10)

    tr = subs.add_parser("train", help="Fit feature selection + XGBoost using training patients ONLY")
    tr.add_argument("--train-csv", required=True)
    tr.add_argument("--valid-csv", default=None)
    tr.add_argument("--outdir", required=True)
    tr.add_argument("--label-col", default="label")
    tr.add_argument("--top-k", type=int, default=30)
    tr.add_argument("--threshold", type=float, default=0.5)
    tr.add_argument("--seed", type=int, default=42)

    pred = subs.add_parser("predict", help="Predict with a saved/frozen XGBoost pipeline")
    pred.add_argument("--csv", required=True)
    pred.add_argument("--model", required=True)
    pred.add_argument("--out", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.command == "extract":
        extract_table(args.manifest, args.out, args.pet_bin_width,
                      args.ct_bin_width, args.min_voxels)
    elif args.command == "train":
        train_model(args.train_csv, args.valid_csv, args.outdir,
                    args.label_col, args.top_k, args.threshold, args.seed)
    else:
        predict_model(args.csv, args.model, args.out)


if __name__ == "__main__":
    main()
