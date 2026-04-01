from __future__ import annotations

import math
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Make parent directory importable
dirname = pathlib.Path(__file__).resolve().parent
sys.path.append(str(dirname / ".."))

from pyopmnearwell.ml import nn
from pyopmnearwell.utils import units
from runspecs import runspecs_ensemble, trainspecs


def build_shaped_stencil_features(
    raw_features: np.ndarray,
    trainspecs: dict,
    stencil_size: int = 3,
) -> np.ndarray:
    """
    Rebuild the same stencil features as in restructure_data(),
    but KEEP the shaped tensor:
        (nmembers, nt, nlayers, nx, nfeat)
    """

    new_features_lst: list[np.ndarray] = []

    # Local stencil features: pressure (0), saturation (1)
    for i in [0, 1]:
        feature = raw_features[..., i].copy()

        if i == 0:
            if trainspecs["pressure_unit"] == "bar":
                feature = feature * units.PASCAL_TO_BAR

            if trainspecs["pressure_padding"] == "zeros":
                padding_mode = "constant"
                padding_value = 0.0
            else:  # neighbor
                padding_mode = "edge"
                padding_value = 0.0

        else:  # saturation
            if trainspecs["saturation_padding"] == "zeros":
                padding_mode = "constant"
                padding_value = 0.0
            else:
                padding_mode = "edge"
                padding_value = 0.0

        half = math.floor(stencil_size / 2)

        if padding_mode == "constant":
            upper_features = [
                np.pad(
                    feature[:, :, :-(j + 1), ...],
                    [(0, 0) if k != 2 else ((j + 1), 0) for k in range(feature.ndim)],
                    mode=padding_mode,
                    constant_values=padding_value,
                )
                for j in range(half)
            ]
            lower_features = [
                np.pad(
                    feature[:, :, (j + 1):, ...],
                    [(0, 0) if k != 2 else (0, (j + 1)) for k in range(feature.ndim)],
                    mode=padding_mode,
                    constant_values=padding_value,
                )
                for j in range(half)
            ]
        else:
            upper_features = [
                np.pad(
                    feature[:, :, :-(j + 1), ...],
                    [(0, 0) if k != 2 else ((j + 1), 0) for k in range(feature.ndim)],
                    mode=padding_mode,
                )
                for j in range(half)
            ]
            lower_features = [
                np.pad(
                    feature[:, :, (j + 1):, ...],
                    [(0, 0) if k != 2 else (0, (j + 1)) for k in range(feature.ndim)],
                    mode=padding_mode,
                )
                for j in range(half)
            ]

        new_features_lst.extend(upper_features + [feature] + lower_features)

    # Raw feature indices from your current upscale pipeline
    # pressure, saturation, radius, FGIT, WGIR, TSLSI, LSID, time_days, PI
    RADIUS_IDX = 2
    FGIT_IDX = 3
    WGIR_IDX = 4
    TSLSI_IDX = 5
    LSID_IDX = 6
    TIME_IDX = 7
    PI_IDX = 8

    # Add global / engineered features
    new_features_lst.append(raw_features[..., RADIUS_IDX])   # radius
    new_features_lst.append(raw_features[..., FGIT_IDX])     # total_injected_volume
    new_features_lst.append(raw_features[..., WGIR_IDX])     # injection_rate
    new_features_lst.append(raw_features[..., TSLSI_IDX])    # time_since_last_shut_in
    new_features_lst.append(raw_features[..., LSID_IDX])     # last_shut_in_duration
    # time_days is NOT selected in your current trainspecs, so we do not need it here

    PI = raw_features[..., PI_IDX]
    eps = 1e-12
    if trainspecs["WI_log"]:
        PI_safe = np.where(np.isfinite(PI) & (PI > 0), PI, 1.0)
        new_features_lst.append(np.log10(np.maximum(PI_safe, eps)))
    else:
        new_features_lst.append(np.nan_to_num(PI, nan=0.0, posinf=0.0, neginf=0.0))

    all_features = np.stack(new_features_lst, axis=-1)

    # Match current trainspecs["features"] ordering exactly
    feature_to_index = {
        "pressure_upper": 0,
        "pressure": 1,
        "pressure_lower": 2,
        "saturation_upper": 3,
        "saturation": 4,
        "saturation_lower": 5,
        "radius": 6,
        "total_injected_volume": 7,
        "injection_rate": 8,
        "time_since_last_shut_in": 9,
        "last_shut_in_duration": 10,
        #"time_days": 10,
        "PI_analytical": 11,
    }

    selected = all_features[..., [feature_to_index[f] for f in trainspecs["features"]]]
    return selected.astype(np.float32)

def plot_member_wi_vs_time(
    raw_features: np.ndarray,
    raw_targets: np.ndarray,
    model: keras.Model,
    nn_dir: pathlib.Path,
    member: int = 0,
    radius_index: int = 3,
    savepath: pathlib.Path | None = None,
) -> None:
    """
    Plot true WI (dots) and predicted WI (line) vs time for each layer,
    at one fixed radius index, for one member.
    """

    X_shaped = build_shaped_stencil_features(raw_features, trainspecs, stencil_size=3)

    WGIR_IDX = 4

    X_member = X_shaped[member]
    y_member = raw_targets[member]
    q_member = raw_features[member, ..., WGIR_IDX]

    saved_shape = list(X_member.shape)
    X_flat = X_member.reshape(-1, saved_shape[-1])

    q_mask = q_member > 0.0
    q_mask_flat = q_mask.reshape(-1)

    y_pred_flat = np.full(X_flat.shape[0], np.nan, dtype=np.float32)

    if np.any(q_mask_flat):
        X_eval = X_flat[q_mask_flat]
        y_pred_log_eval = nn.scale_and_evaluate(
            model,
            X_eval,
            nn_dir / "scalings.csv",
        ).numpy().reshape(-1)

        if trainspecs["WI_log"]:
            y_pred_eval = 10 ** y_pred_log_eval
        else:
            y_pred_eval = y_pred_log_eval

        y_pred_flat[q_mask_flat] = y_pred_eval.astype(np.float32)

    y_pred = y_pred_flat.reshape(saved_shape[:-1])

    y_true = y_member.copy()

    reportstep_length = 0.5  # s################################set manually######################################
    nt = X_member.shape[0]
    x_values = np.arange(nt) * reportstep_length
    
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = plt.cm.Blues(np.linspace(1, 0.5, y_true.shape[1]))

    for layer, color in zip(range(y_true.shape[1]), colors):
        y_true_layer = y_true[:, layer, radius_index]
        y_pred_layer = y_pred[:, layer, radius_index]
        q_layer = q_member[:, layer, radius_index]

        q_valid = q_layer > 0.0
        target_valid = np.isfinite(y_true_layer) & (y_true_layer > 0)
        valid = q_valid & target_valid

        y_true_plot = np.where(valid, y_true_layer, np.nan)
        y_pred_plot = np.where(valid, y_pred_layer, np.nan)

        ax.scatter(x_values, y_true_plot, color=color, s=18, label=f"layer {layer}: WI data")
        ax.plot(x_values, y_pred_plot, color=color, linewidth=1.8, label=f"layer {layer}: WI NN")

    ax.set_xlabel("Time [d]")
    ax.set_ylabel(r"WI [$m^4\,s/kg$]")
    ax.set_title(f"WI vs time for member {member} at radius index {radius_index}")
    ax.set_xlim(-1, x_values[-1] + 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.72, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if savepath is None:
        savepath = nn_dir / f"member_{member}_WI_vs_time_radius_{radius_index}.png"

    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    data_dir = dirname / "dataset_ex"
    nn_dir = dirname / "nn"

    ds = tf.data.Dataset.load(str(data_dir))
    raw_features, raw_targets = next(iter(ds.batch(batch_size=len(ds)).as_numpy_iterator()))

    print("raw_features shape:", raw_features.shape)
    print("raw_targets shape :", raw_targets.shape)

    model = keras.models.load_model(nn_dir / "bestmodel.keras")

    plot_dir = nn_dir / "late_members_ex"
    plot_dir.mkdir(parents=True, exist_ok=True)

    start_idx = max(raw_features.shape[0] - 30, 0)

    for member in range(start_idx, raw_features.shape[0]):
        plot_member_wi_vs_time(
            raw_features=raw_features,
            raw_targets=raw_targets,
            model=model,
            nn_dir=nn_dir,
            member=member,
            radius_index=3,
            savepath=plot_dir / f"member_{member}_WI_vs_time_radius_3.png",
        )