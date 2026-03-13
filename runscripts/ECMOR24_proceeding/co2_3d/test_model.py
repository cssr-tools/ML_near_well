import argparse
import pathlib
import csv
import numpy as np
import tensorflow as tf


def read_target_scaling(scalings_csv: pathlib.Path, target_name: str = "output_WI"):
    """Returnerer (y_min, y_max, a, b) der target_range er [a,b]."""
    y_min = y_max = None
    a, b = -1.0, 1.0  # default

    with scalings_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            var = row["variable"]
            if var == target_name:
                y_min = float(row["min"])
                y_max = float(row["max"])
            elif var == "target_range":
                a = float(row["min"])
                b = float(row["max"])

    if y_min is None or y_max is None:
        raise ValueError(f"Fant ikke {target_name} i {scalings_csv}")
    return y_min, y_max, a, b


def main(nn_dir: str):
    nn_dir = pathlib.Path(nn_dir)

    model_path = nn_dir / "bestmodel.keras"
    test_path = nn_dir / "testset_scaled.npz"
    scalings_path = nn_dir / "scalings.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Fant ikke: {model_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Fant ikke: {test_path}")
    if not scalings_path.exists():
        raise FileNotFoundError(f"Fant ikke: {scalings_path}")

    model = tf.keras.models.load_model(model_path, compile=False)

    data = np.load(test_path)
    Xs = data["X"]
    ys = data["y"]

    yhat_s = model.predict(Xs, verbose=0)

    # --- Scaled metrics ---
    rmse_s = float(np.sqrt(np.mean((yhat_s - ys) ** 2)))
    mae_s = float(np.mean(np.abs(yhat_s - ys)))

    # --- Convert scaled error to WI units using scalings.csv ---
    y_min, y_max, a, b = read_target_scaling(scalings_path, target_name="output_WI")
    wi_range = (y_max - y_min)
    scaled_span = (b - a)  # typisk 2 når [-1,1]

    factor = wi_range / scaled_span  # ΔWI = Δscaled * (range/span)
    rmse_wi = rmse_s * factor
    mae_wi = mae_s * factor

    print(f"Scaled RMSE: {rmse_s}")
    print(f"Scaled MAE : {mae_s}")
    print(f"WI min/max : {y_min} .. {y_max}  (range={wi_range})")
    print(f"RMSE (WI)  : {rmse_wi}")
    print(f"MAE  (WI)  : {mae_wi}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn_dir", required=True, help="Mappe med bestmodel.keras, testset_scaled.npz og scalings.csv")
    args = parser.parse_args()
    main(args.nn_dir)