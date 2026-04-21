from __future__ import annotations

import math
import pathlib
from typing import Any
import csv

import numpy as np
import tensorflow as tf
from pyopmnearwell.ml import ensemble
from pyopmnearwell.utils import units

dirname: pathlib.Path = pathlib.Path(__file__).parent

# TODO: Generalize this for different stencils.
FEATURE_TO_INDEX = {
    "pressure_upper": 0,
    "pressure": 1,
    "pressure_lower": 2,
    "saturation_upper": 3,
    "saturation": 4,
    "saturation_lower": 5,
    "radius": 6,
    "total_injected_volume": 7,
    "injection_rate": 8,
    "current_injection_time": 9,
    "previous_shutin_time": 10,
    "previous_injection_time": 11,
    "older_history_time": 12,
    "PI_analytical": 13,
}


plotted_values_units: dict[str, str] = {
    "WI": r"[m^4 \cdot s/kg]",
    "bhp": "[Pa]",
    "perm": "[mD]",
}
x_axis_units: dict[str, str] = {"time": "[d]", "radius": "[m]"}
comparisons_inverse: dict[str, str] = {
    "timesteps": "layer",
    "layers": "timestep or radius",
}



def restructure_data(
    data_dirname: str | pathlib.Path,
    new_data_dirname: str | pathlib.Path,
    trainspecs: dict[str, Any],
    stencil_size: int = 3,
) -> None:
    """_summary_

    The final dataset will be in a flattened shape.

    Note: The local features inside the stencil always (!) range FROM upper TO lower
        cells.

    TODO: Generalize this for different stencils.
    The new features are in the following order:
    1. PRESSURE - upper neighbor
    2. PRESSURE - cell
    3. PRESSURE - lower neighbor
    4. SATURATION - upper neighbor
    5. SATURATION - cell
    6. SATURATION - lower neighbor
    10. radius
    11. Injection rate 
    12. total injected gas
    13. analytical PI

    Args:
        data_dirname (str | pathlib.Path): _description_
        stencil_size (int, optional): _description_. Defaults to 3.

    """
    # Load data.
    ds: tf.data.Dataset = tf.data.Dataset.load(str(data_dirname))
    features, targets = next(iter(ds.batch(batch_size=len(ds)).as_numpy_iterator()))
    print("UPSCALE nfeat:", features.shape[-1])
    print("UPSCALE tslsi first 5:", features[0, :5, 0, 0, 5])
    print("UPSCALE last shut first 5:", features[0, :5, 0, 0, 6])
    print("UPSCALE time first 5:", features[0, :5, 0, 0, 7])    # Add upper and lower cell features to create the training data for the stencil.
    new_features_lst: list[np.ndarray] = []

    # ---- ONLY stencil local features: pressure (0) and saturation (1) ----
    for i in [0, 1]:
        feature: np.ndarray = features[..., i]

        # Pad all local features and scale values.
        if i == 0:
            if trainspecs["pressure_unit"] == "bar":
                feature = feature * units.PASCAL_TO_BAR

            if trainspecs["pressure_padding"] == "zeros":
                padding_mode: str = "constant"
                padding_value: float = 0.0
            elif trainspecs["pressure_padding"] == "neighbor":
                padding_mode = "edge"
                padding_value = 0.0

        elif i == 1:
            if trainspecs["saturation_padding"] == "zeros":
                padding_mode = "constant"
                padding_value = 0.0
            else:
                padding_mode = "edge"
                padding_value = 0.0

        # Pad the third (layers) feature dimension.
        if padding_mode == "constant":
            upper_features = [
                np.pad(
                    feature[:, :, : -(j + 1), ...],
                    [(0, 0) if k != 2 else ((j + 1), 0) for k in range(feature.ndim)],
                    mode=padding_mode,
                    constant_values=padding_value,
                )
                for j in range(math.floor(stencil_size / 2))
            ]
            lower_features = [
                np.pad(
                    feature[:, :, (j + 1) :, ...],
                    [(0, 0) if k != 2 else (0, (j + 1)) for k in range(feature.ndim)],
                    mode=padding_mode,
                    constant_values=padding_value,
                )
                for j in range(math.floor(stencil_size / 2))
            ]
        else:
            upper_features = [
                np.pad(
                    feature[:, :, : -(j + 1), ...],
                    [(0, 0) if k != 2 else ((j + 1), 0) for k in range(feature.ndim)],
                    mode=padding_mode,
                )
                for j in range(math.floor(stencil_size / 2))
            ]
            lower_features = [
                np.pad(
                    feature[:, :, (j + 1) :, ...],
                    [(0, 0) if k != 2 else (0, (j + 1)) for k in range(feature.ndim)],
                    mode=padding_mode,
                )
                for j in range(math.floor(stencil_size / 2))
            ]

        new_features_lst.extend(upper_features + [feature] + lower_features)

    # ---- Add back global features using explicit indices in UPSCALER features tensor ----
    RADIUS_IDX        = 2
    FGIT_IDX          = 3
    WGIR_IDX          = 4
    CURR_INJ_IDX      = 5
    PREV_SHUTIN_IDX   = 6
    PREV_INJ_IDX      = 7
    OLDER_HIST_IDX    = 8
    TIME_IDX          = 9
    PI_IDX            = 10

    new_features_lst.append(features[..., RADIUS_IDX])       # radius
    new_features_lst.append(features[..., FGIT_IDX])         # total_injected_volume
    new_features_lst.append(features[..., WGIR_IDX])         # injection_rate
    new_features_lst.append(features[..., CURR_INJ_IDX])     # current_injection_time
    new_features_lst.append(features[..., PREV_SHUTIN_IDX])  # previous_shutin_time
    new_features_lst.append(features[..., PREV_INJ_IDX])     # previous_injection_time
    new_features_lst.append(features[..., OLDER_HIST_IDX])   # older_history_time
    # new_features_lst.append(features[..., TIME_IDX])       # time_days, only if you want it

    PI = features[..., PI_IDX]
    eps = 1e-12

    if trainspecs["WI_log"]:
        PI_safe = np.where(np.isfinite(PI) & (PI > 0), PI, 1.0)
        new_features_lst.append(np.log10(np.maximum(PI_safe, eps)))
    else:
        new_features_lst.append(np.nan_to_num(PI, nan=0.0, posinf=0.0, neginf=0.0))

    # --- Build final feature tensor ---
    new_features = np.stack(new_features_lst, axis=-1)

################ENDRET! sparer kun på de radene som inneholder WI#############
    # --- Select chosen features --- 
    new_features = new_features[..., [FEATURE_TO_INDEX[f] for f in trainspecs["features"]]]

    # --- Flatten ---
    X = new_features.reshape(-1, new_features.shape[-1])   # (N, F)
    y = targets.reshape(-1)                                # (N,)

    # --- NaN-safe target transform + drop unlabeled (shut-in) ---
    valid = np.isfinite(y)
    if trainspecs["WI_log"]:
        valid &= (y > 0)
        y_safe = np.where(valid, y, 1.0)
        y_out = np.log10(np.maximum(y_safe, eps))
    else:
        y_out = np.where(valid, y, 0.0)

    X = X[valid]
    y_out = y_out[valid]
    
    ####### ENDRET 16.03 ####################################
    nmembers, nt, nlayers, nx, _ = new_features.shape

    member_ids = np.broadcast_to(
        np.arange(nmembers)[:, None, None, None],
        (nmembers, nt, nlayers, nx),
    ).reshape(-1)

    time_ids = np.broadcast_to(
        np.arange(nt)[None, :, None, None],
        (nmembers, nt, nlayers, nx),
    ).reshape(-1)

    layer_ids = np.broadcast_to(
        np.arange(nlayers)[None, None, :, None],
        (nmembers, nt, nlayers, nx),
    ).reshape(-1)

    x_ids = np.broadcast_to(
        np.arange(nx)[None, None, None, :],
        (nmembers, nt, nlayers, nx),
    ).reshape(-1)

    X = new_features.reshape(-1, new_features.shape[-1])
    y = targets.reshape(-1)

    valid = np.isfinite(y)
    if trainspecs["WI_log"]:
        valid &= (y > 0)
        y_safe = np.where(valid, y, 1.0)
        y_out = np.log10(np.maximum(y_safe, eps))
    else:
        y_out = np.where(valid, y, 0.0)

    X = X[valid]
    y_out = y_out[valid]

    member_ids = member_ids[valid]
    time_ids = time_ids[valid]
    layer_ids = layer_ids[valid]
    x_ids = x_ids[valid]

    new_data_dirname = pathlib.Path(new_data_dirname)
    new_data_dirname.mkdir(parents=True, exist_ok=True)

    with (new_data_dirname / "row_to_run_map.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_idx", "member_id", "time_id", "layer_id", "x_id"])
        for i, m, t, l, x in zip(
            np.arange(len(member_ids)), member_ids, time_ids, layer_ids, x_ids
        ):
            writer.writerow([i, m, t, l, x])

    ####### ENDRET 16.03 ####################################


    ensemble.store_dataset(
        X.astype(np.float32),
        y_out[..., None].astype(np.float32),
        new_data_dirname,
    )


def restructure_data_sequence(
    data_dirname: str | pathlib.Path,
    new_data_dirname: str | pathlib.Path,
    trainspecs: dict[str, Any],
    stencil_size: int = 3,
) -> None:
  
    ds: tf.data.Dataset = tf.data.Dataset.load(str(data_dirname))
    features, targets = next(iter(ds.batch(batch_size=len(ds)).as_numpy_iterator()))

    print("RAW features shape:", features.shape)
    print("RAW targets  shape:", targets.shape)

    # ------------------------------------------------------------------
    # 1. Build stencil features exactly like in the flat FCNN pipeline
    # ------------------------------------------------------------------
    new_features_lst: list[np.ndarray] = []

    # Local stencil features: pressure (0), saturation (1)
    for i in [0, 1]:
        feature: np.ndarray = features[..., i]

        if i == 0:
            if trainspecs["pressure_unit"] == "bar":
                feature = feature * units.PASCAL_TO_BAR

            if trainspecs["pressure_padding"] == "zeros":
                padding_mode = "constant"
                padding_value = 0.0
            elif trainspecs["pressure_padding"] == "neighbor":
                padding_mode = "edge"
                padding_value = 0.0
            else:
                raise ValueError(
                    f"Unknown pressure_padding={trainspecs['pressure_padding']}"
                )

        elif i == 1:
            if trainspecs["saturation_padding"] == "zeros":
                padding_mode = "constant"
                padding_value = 0.0
            elif trainspecs["saturation_padding"] == "neighbor":
                padding_mode = "edge"
                padding_value = 0.0
            else:
                raise ValueError(
                    f"Unknown saturation_padding={trainspecs['saturation_padding']}"
                )

        half = math.floor(stencil_size / 2)

        if padding_mode == "constant":
            upper_features = [
                np.pad(
                    feature[:, :, : -(j + 1), ...],
                    [(0, 0) if k != 2 else ((j + 1), 0) for k in range(feature.ndim)],
                    mode=padding_mode,
                    constant_values=padding_value,
                )
                for j in range(half)
            ]
            lower_features = [
                np.pad(
                    feature[:, :, (j + 1) :, ...],
                    [(0, 0) if k != 2 else (0, (j + 1)) for k in range(feature.ndim)],
                    mode=padding_mode,
                    constant_values=padding_value,
                )
                for j in range(half)
            ]
        else:
            upper_features = [
                np.pad(
                    feature[:, :, : -(j + 1), ...],
                    [(0, 0) if k != 2 else ((j + 1), 0) for k in range(feature.ndim)],
                    mode=padding_mode,
                )
                for j in range(half)
            ]
            lower_features = [
                np.pad(
                    feature[:, :, (j + 1) :, ...],
                    [(0, 0) if k != 2 else (0, (j + 1)) for k in range(feature.ndim)],
                    mode=padding_mode,
                )
                for j in range(half)
            ]

        new_features_lst.extend(upper_features + [feature] + lower_features)

    # ------------------------------------------------------------------
    # 2. Add global / engineered features
    # ------------------------------------------------------------------
    RADIUS_IDX        = 2
    FGIT_IDX          = 3
    WGIR_IDX          = 4
    CURR_INJ_IDX      = 5
    PREV_SHUTIN_IDX   = 6
    PREV_INJ_IDX      = 7
    OLDER_HIST_IDX    = 8
    TIME_IDX          = 9
    PI_IDX            = 10

    new_features_lst.append(features[..., RADIUS_IDX])       # radius
    new_features_lst.append(features[..., FGIT_IDX])         # total_injected_volume
    new_features_lst.append(features[..., WGIR_IDX])         # injection_rate
    new_features_lst.append(features[..., CURR_INJ_IDX])     # current_injection_time
    new_features_lst.append(features[..., PREV_SHUTIN_IDX])  # previous_shutin_time
    new_features_lst.append(features[..., PREV_INJ_IDX])     # previous_injection_time
    new_features_lst.append(features[..., OLDER_HIST_IDX])   # older_history_time
    # new_features_lst.append(features[..., TIME_IDX])       # time_days, only if used

    PI = features[..., PI_IDX]
    eps = 1e-12

    if trainspecs["WI_log"]:
        PI_safe = np.where(np.isfinite(PI) & (PI > 0), PI, 1.0)
        new_features_lst.append(np.log10(np.maximum(PI_safe, eps)))
    else:
        new_features_lst.append(np.nan_to_num(PI, nan=0.0, posinf=0.0, neginf=0.0))

    new_features = np.stack(new_features_lst, axis=-1)

    # Keep only selected features, same as FCNN pipeline
    new_features = new_features[
        ..., [FEATURE_TO_INDEX[f] for f in trainspecs["features"]]
    ]

    print("Stencil feature tensor shape:", new_features.shape)

    # ------------------------------------------------------------------
    # 3. Transform targets, but KEEP time axis
    # ------------------------------------------------------------------
    valid = np.isfinite(targets)
    if trainspecs["WI_log"]:
        valid &= (targets > 0)
        targets_safe = np.where(valid, targets, 1.0)
        y_all = np.log10(np.maximum(targets_safe, eps))
    else:
        y_all = np.where(valid, targets, 0.0)

    # mask = 1 where target is valid, 0 during shut-in / unlabeled rows
    mask_all = valid.astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Collapse spatial dims into sequence batch dimension
    # ------------------------------------------------------------------
    # Current shape:
    #   X: (nruns, nt, nlayers, nxcells, nfeat)
    #   y: (nruns, nt, nlayers, nxcells)
    #
    # Want:
    #   X_seq: (nruns*nlayers*nxcells, nt, nfeat)
    #   y_seq: (nruns*nlayers*nxcells, nt, 1)
    #   m_seq: (nruns*nlayers*nxcells, nt, 1)

    nruns, nt, nlayers, nxcells, nfeat = new_features.shape

    X_seq = np.transpose(new_features, (0, 2, 3, 1, 4)).reshape(
        nruns * nlayers * nxcells, nt, nfeat
    )

    y_seq = np.transpose(y_all, (0, 2, 3, 1)).reshape(
        nruns * nlayers * nxcells, nt, 1
    )

    mask_seq = np.transpose(mask_all, (0, 2, 3, 1)).reshape(
        nruns * nlayers * nxcells, nt, 1
    )

    print("Sequence X shape:", X_seq.shape)
    print("Sequence y shape:", y_seq.shape)
    print("Sequence mask shape:", mask_seq.shape)

    # ------------------------------------------------------------------
    # 5. Optional: remove sequences with no valid targets at all
    # ------------------------------------------------------------------
    keep_seq = np.any(mask_seq[..., 0] > 0, axis=1)

    X_seq = X_seq[keep_seq]
    y_seq = y_seq[keep_seq]
    mask_seq = mask_seq[keep_seq]

    print("Filtered sequence X shape:", X_seq.shape)
    print("Filtered sequence y shape:", y_seq.shape)
    print("Filtered sequence mask shape:", mask_seq.shape)


    ######ENDRET 16.03 ################################

    # Lag mapping fra flattenet rad til original indeks
    nmembers, nt, nlayers, nx, _ = new_features.shape

    member_ids = np.broadcast_to(
        np.arange(nmembers)[:, None, None, None],
        (nmembers, nt, nlayers, nx),
    ).reshape(-1)

    time_ids = np.broadcast_to(
        np.arange(nt)[None, :, None, None],
        (nmembers, nt, nlayers, nx),
    ).reshape(-1)

    layer_ids = np.broadcast_to(
        np.arange(nlayers)[None, None, :, None],
        (nmembers, nt, nlayers, nx),
    ).reshape(-1)

    x_ids = np.broadcast_to(
        np.arange(nx)[None, None, None, :],
        (nmembers, nt, nlayers, nx),
    ).reshape(-1)

    row_ids = np.arange(member_ids.shape[0])

    new_data_dirname = pathlib.Path(new_data_dirname)
    new_data_dirname.mkdir(parents=True, exist_ok=True)

    with (new_data_dirname / "row_to_run_map.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_idx", "member_id", "time_id", "layer_id", "x_id"])
        for row in zip(row_ids, member_ids, time_ids, layer_ids, x_ids):
            writer.writerow(row)
    ######ENDRET 16.03 ################################
    # ------------------------------------------------------------------
    # 6. Store dataset
    # ------------------------------------------------------------------
    # First simple version: store only X and y.
    # Invalid timesteps remain in y as 0.0, and you must use mask_seq later
    # during training to avoid learning from shut-in targets.
    #
    # If ensemble.store_dataset only supports (features, targets), do this:
    ensemble.store_dataset(
        X_seq.astype(np.float32),
        y_seq.astype(np.float32),
        new_data_dirname,
    )

    # If you want, we can also store mask_seq separately as .npy:
    np.save(pathlib.Path(new_data_dirname) / "mask.npy", mask_seq.astype(np.float32))