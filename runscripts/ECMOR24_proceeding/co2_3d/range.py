from __future__ import annotations

import csv
import pathlib
import numpy as np
import tensorflow as tf

from WIplot_ex import build_shaped_stencil_features
from runspecs import trainspecs

dirname = pathlib.Path(__file__).parent

data_dir = dirname / "dataset_ex"
nn_dir = dirname / "nn"
scalings_csv = nn_dir / "scalings.csv"


def read_feature_ranges(csv_path: pathlib.Path) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            var = row["variable"]
            try:
                vmin = float(row["min"])
                vmax = float(row["max"])
            except ValueError:
                continue
            ranges[var] = (vmin, vmax)
    return ranges


def main():
    ds = tf.data.Dataset.load(str(data_dir))
    raw_features, raw_targets = next(iter(ds.batch(batch_size=len(ds)).as_numpy_iterator()))

    X = build_shaped_stencil_features(raw_features, trainspecs, stencil_size=3)
    feature_names = list(trainspecs["features"])
    train_ranges = read_feature_ranges(scalings_csv)

    print(f"X shape: {X.shape}")
    print(f"Using scalings from: {scalings_csv}")
    print()

    for i, name in enumerate(feature_names):
        vals = X[..., i]
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            print(f"{name:25s} : no finite values")
            continue

        data_min = float(np.min(vals))
        data_max = float(np.max(vals))

        scaling_name = f"input_{name}"

        if scaling_name not in train_ranges:
            print(
                f"{name:25s} : NOT FOUND as {scaling_name} in scalings.csv "
                f"| data=[{data_min:.6g}, {data_max:.6g}]"
            )
            continue

        train_min, train_max = train_ranges[scaling_name]

        below = int(np.sum(vals < train_min))
        above = int(np.sum(vals > train_max))
        total = int(vals.size)
        out = below + above
        frac = out / total

        status = "OK" if out == 0 else "OUT-OF-RANGE"

        print(
            f"{name:25s} : "
            f"train=[{train_min:.6g}, {train_max:.6g}] "
            f"data=[{data_min:.6g}, {data_max:.6g}] "
            f"below={below} above={above} out={out}/{total} ({100*frac:.2f}%) "
            f"{status}"
        )


if __name__ == "__main__":
    main()