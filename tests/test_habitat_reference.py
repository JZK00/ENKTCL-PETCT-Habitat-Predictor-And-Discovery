import numpy as np
import pandas as pd
import pytest

from habitat_reference import (FEATURES, assign_habitats, fit_habitats,
                               patient_fractions, read_descriptors)


def sample_data():
    rng = np.random.default_rng(1)
    centers = [[12, 3, 35, 2.5], [2, 0.5, 120, 0.7], [6, 2.0, 50, 3.0]]
    rows = []
    for i, center in enumerate(centers):
        for j in range(15):
            rows.append([f"P{j % 3}", f"{i}-{j}", j + 1,
                         *rng.normal(center, [0.2, 0.1, 2, 0.1])])
    return pd.DataFrame(rows, columns=["patient_id", "supervoxel_id",
                                       "voxel_count", *FEATURES])


def test_frozen_inference_and_voxel_weighting():
    data = sample_data()
    model = fit_habitats(data)
    original_centroids = model["kmeans"].cluster_centers_.copy()
    assigned = assign_habitats(data, model)
    assert set(assigned.habitat) == {1, 2, 3}
    assert np.array_equal(original_centroids, model["kmeans"].cluster_centers_)
    fractions = patient_fractions(assigned)
    assert np.allclose(fractions.iloc[:, 1:].sum(axis=1), 1)
    raw_habitat_1 = next(k for k, v in model["mapping"].items() if v == 1)
    assert raw_habitat_1 == int(np.argmax(model["centroids"][:, 0]))


def test_invalid_inputs(tmp_path):
    frame = sample_data().drop(columns=["ct_entropy"])
    csv = tmp_path / "bad.csv"
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Missing columns"):
        read_descriptors(csv)
    with pytest.raises(ValueError, match="Mapping"):
        fit_habitats(sample_data(), mapping={0: 1, 1: 1, 2: 3})
