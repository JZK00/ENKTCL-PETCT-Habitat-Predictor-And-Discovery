# ENKTCL-PETCT-Habitat-Predictor-And-Discovery
Interpretable PET/CT Habitat Analysis of Intratumoral Heterogeneity Predicts Response and Stratification in Limited-stage Extranodal NK/T-cell Lymphoma


## Study overview

This retrospective study included 366 patients with limited-stage ENKTCL from four centres, pooled into a **training cohort (n=252)** and a **validation cohort (n=114)**. The PET/CT workflow comprises registration and resampling (1-mm isotropic), 9 × 9 × 9 voxel supervoxel partitioning, four PET/CT descriptors (PET SUV, PET local entropy, CT attenuation and CT local entropy), population-level K-means habitat discovery (three habitats), habitat-specific radiomics, harmonisation, feature selection, XGBoost modeling, and clinical analysis.

Habitat 1 was a small metabolically active, heterogeneous compartment (5.4% of tumour voxels) with an AUC of 0.823/0.824 for treatment non-response in the training/validation cohorts.

## What is provided here

`habitat_reference.py` is a small **runnable reference of the *descriptor-level clustering and frozen assignment* stage**, plus a synthetic-data demo. It fits a `StandardScaler` and 3-cluster K-means on **training descriptors only**, saves both models, assigns held-out supervoxels without refitting, and exports voxel-weighted patient habitat fractions. Tests are in `tests/`.

**Important scope:** This repository does **not** yet contain the original DICOM/NIfTI preprocessing and segmentation, local entropy/supervoxel extraction, PyRadiomics/ComBat pipeline, ICC screening, SHAP-XGBoost trained model, nomogram, figures, data, or the original fitted model weights. The reference code therefore **cannot reproduce the manuscript AUCs or RadScoreHabitat**. Do not interpret its demo as a clinical predictor. 

The original habitat workflow builds on [HABIT](https://github.com/lichao312214129/HABIT).

## Installation and quick start

```bash
python -m venv .venv
# Activate your virtual environment first
pip install -r requirements.txt
python habitat_reference.py demo --dir demo_output
```

The demo uses **synthetic** descriptors and writes `validation_supervoxels.csv`, `validation_patients.csv`, and a saved model inside `demo_output/`.

For precomputed real *training-only* supervoxel descriptors, prepare a CSV with columns:

```
patient_id,supervoxel_id,voxel_count,pet_suv,pet_entropy,ct_intensity,ct_entropy
```

Each row represents one tumour supervoxel. `voxel_count` is the number of tumour voxels in that supervoxel (needed for correct habitat-volume proportions). SUV and HU should retain their physical units; the four descriptors are standardized **only for clustering**, with parameters learned on training data.

```bash
python habitat_reference.py train --csv training_descriptors.csv --model output/habitat_model.joblib
python habitat_reference.py predict --csv validation_descriptors.csv --model output/habitat_model.joblib --out output/validation
```

The training command displays the original-scale K-means centroids and suggested Habitat 1/2/3 correspondence. **Check this correspondence against the study's original centroid profiles.** Labels are arbitrary in K-means; if needed, manually specify the validated mapping at training time, e.g. `--mapping 0:2,1:1,2:3`. A different mapping changes habitat IDs and must not be silently substituted for the reported study. The validation set must not be used to choose the number of clusters, centroids, scaler, or mapping.

## Extract features (by using PyRadiomics)

```bash
python habitat_radiomics_xgboost.py extract --manifest train_manifest.csv --out train_radiomics.csv
python habitat_radiomics_xgboost.py extract --manifest external_manifest.csv --out external_radiomics.csv
```

For every habitat and PET/CT modality, the example calculates Original-image first-order and texture features (GLCM, GLRLM, GLSZM, GLDM, NGTDM). Shape features are calculated once from PET and the same mask. Every output row is one patient, with columns such as `PET_H1_original_firstorder_Mean`, `CT_H2_original_glcm_Contrast`. Habitats with fewer than 10 voxels are skipped, yielding missing feature values.

Default fixed bin widths are **PET SUV = 0.5** and **CT HU = 25**, as **illustrative settings only**, not the final verified settings. Adjust with `--pet-bin-width` / `--ct-bin-width`. No normalization or resampling is performed inside PyRadiomics. Match the manuscript's voxel spacing, binning, filtering, preprocessing, and extraction version before scientific use. For fair external validation, PET/CT and masks must have been aligned using a predefined, consistent preprocessing pipeline.

## Train XGBoost using training patients only

```bash
python habitat_radiomics_xgboost.py train --train-csv train_radiomics.csv --valid-csv external_radiomics.csv --outdir results --label-col label --top-k 30
```

The script excludes patient IDs and mask volume counts from predictors, drops all-missing/constant columns on the *training* cohort, fits median imputation and univariate top-k feature selection **only on training patients**, and then fits XGBoost. It applies this frozen pipeline to external data; it prevents overlapping patient IDs between cohorts.

Outputs: `results/model.joblib`, `feature_importance.csv`, `train_predictions.csv`, `validation_predictions.csv`, `metrics.json` (binary AUC, accuracy at illustrative 0.5 threshold). XGBoost feature importances are descriptive, not SHAP explanations. Model scores are uncalibrated probabilities and should not be interpreted as validated clinical risk estimates.

## Predict new patient features

```bash
python habitat_radiomics_xgboost.py predict --csv new_radiomics.csv --model results/model.joblib --out results/new_predictions.csv
```

## Reproducibility and data availability

Clinical PET/CT and follow-up records are not distributed here because of patient privacy and institutional restrictions.

The `.gitignore` excludes common medical-image, patient-table, and model-artifact formats to reduce accidental uploads. Only load trusted `.joblib` model files, as joblib uses Python pickle serialization.

## License

See [LICENSE](LICENSE).
