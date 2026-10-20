"""Minimal reference implementation of training-only PET/CT habitat clustering.

Input: one row per tumour supervoxel with four precomputed descriptors.
This is NOT the original end-to-end image-to-RadScore model.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES = ("pet_suv", "pet_entropy", "ct_intensity", "ct_entropy")
REQUIRED = ("patient_id", "supervoxel_id", "voxel_count", *FEATURES)


def read_descriptors(path):
    df = pd.read_csv(path)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty or df[list(REQUIRED)].isna().any().any():
        raise ValueError("Input is empty or contains missing values")
    if df.duplicated(["patient_id", "supervoxel_id"]).any():
        raise ValueError("Duplicate patient_id/supervoxel_id rows")
    values = df[list(FEATURES) + ["voxel_count"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (df["voxel_count"] <= 0).any():
        raise ValueError("Descriptors must be finite and voxel_count > 0")
    return df


def suggested_mapping(centroids):
    """Heuristic only; verify habitat identities against the study centroids."""
    high_suv = int(np.argmax(centroids[:, 0]))
    remaining = [i for i in range(3) if i != high_suv]
    high_ct = max(remaining, key=lambda i: centroids[i, 2])
    last = next(i for i in remaining if i != high_ct)
    return {high_suv: 1, high_ct: 2, last: 3}


def fit_habitats(df, mapping=None, random_state=42):
    """Fit ONLY on training-cohort supervoxels; freeze for validation."""
    x = df.loc[:, FEATURES].to_numpy(dtype=float)
    if len(x) < 3:
        raise ValueError("At least 3 training supervoxels are required")
    scaler = StandardScaler().fit(x)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=random_state).fit(
        scaler.transform(x)
    )
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    if mapping is None:
        mapping = suggested_mapping(centroids)
    if set(mapping.keys()) != {0, 1, 2} or set(mapping.values()) != {1, 2, 3}:
        raise ValueError("Mapping must bijectively assign raw clusters 0,1,2 to habitats 1,2,3")
    return {"scaler": scaler, "kmeans": kmeans, "mapping": mapping,
            "features": FEATURES, "centroids": centroids}


def assign_habitats(df, model):
    """Apply saved training scaler and K-means without refitting."""
    x = df.loc[:, list(model["features"])].to_numpy(dtype=float)
    labels = model["kmeans"].predict(model["scaler"].transform(x))
    result = df.copy()
    result["habitat"] = [model["mapping"][int(i)] for i in labels]
    return result


def patient_fractions(assigned):
    """Voxel-weighted habitat composition, not just supervoxel frequencies."""
    counts = assigned.groupby(["patient_id", "habitat"])["voxel_count"].sum()
    wide = counts.unstack(fill_value=0)
    total = wide.sum(axis=1)
    fractions = pd.DataFrame({"patient_id": wide.index})
    for habitat in (1, 2, 3):
        fractions[f"habitat_{habitat}_fraction"] = (
            wide[habitat].to_numpy() / total.to_numpy()
            if habitat in wide else np.zeros(len(wide))
        )
    return fractions


def write_predictions(df, model, prefix):
    assigned = assign_habitats(df, model)
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(f"{prefix}_supervoxels.csv", index=False)
    patient_fractions(assigned).to_csv(f"{prefix}_patients.csv", index=False)


def demo_data(path, random_state=42):
    """Create SYNTHETIC descriptors to test the interface; no patient data."""
    rng = np.random.default_rng(random_state)
    centers = [(12, 3.2, 35, 2.6), (2.2, 0.6, 130, 0.8), (6, 2.1, 55, 3.1)]
    rows = []
    for patient in range(15):
        for j in range(30):
            c = centers[j % 3]
            feature = rng.normal(c, (0.5, 0.2, 6, 0.2))
            rows.append((f"S{patient:02}", j, 400 + (j % 7), *feature))
    pd.DataFrame(rows, columns=REQUIRED).to_csv(path, index=False)


def parse_mapping(value):
    if value is None:
        return None
    try:
        return dict(tuple(map(int, item.split(":"))) for item in value.split(","))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Use 0:1,1:2,2:3 format") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    train = subs.add_parser("train", help="Fit on TRAINING descriptors only")
    train.add_argument("--csv", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--mapping", type=parse_mapping, default=None)
    train.add_argument("--seed", type=int, default=42)
    pred = subs.add_parser("predict", help="Assign held-out descriptors; never refit")
    pred.add_argument("--csv", required=True)
    pred.add_argument("--model", required=True)
    pred.add_argument("--out", required=True)
    demo = subs.add_parser("demo", help="Generate synthetic data and run end-to-end example")
    demo.add_argument("--dir", default="demo_output")
    args = parser.parse_args()

    if args.command == "demo":
        output = Path(args.dir)
        output.mkdir(parents=True, exist_ok=True)
        csv = output / "synthetic_descriptors.csv"
        demo_data(csv)
        frame = read_descriptors(csv)
        train_df = frame[frame.patient_id < "S10"]
        test_df = frame[frame.patient_id >= "S10"]
        model = fit_habitats(train_df)
        joblib.dump(model, output / "habitat_model.joblib")
        write_predictions(test_df, model, output / "validation")
        print(f"Synthetic example saved in {output}")
    elif args.command == "train":
        frame = read_descriptors(args.csv)
        model = fit_habitats(frame, mapping=args.mapping, random_state=args.seed)
        Path(args.model).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model)
        centers = pd.DataFrame(model["centroids"], columns=FEATURES)
        centers.insert(0, "raw_cluster", np.arange(3))
        centers["habitat"] = centers.raw_cluster.map(model["mapping"])
        print(centers.to_string(index=False))
        print("Verify habitat labels against the manuscript before publication.")
    else:
        # Only load model artifacts from a trusted source: joblib uses pickle.
        model = joblib.load(args.model)
        write_predictions(read_descriptors(args.csv), model, args.out)
        print(f"Wrote {args.out}_supervoxels.csv and {args.out}_patients.csv")


if __name__ == "__main__":
    main()
