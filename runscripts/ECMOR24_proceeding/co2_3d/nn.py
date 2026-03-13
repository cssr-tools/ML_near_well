from __future__ import annotations

import math
import pathlib
from typing import Any

import numpy as np
import tensorflow as tf
from pyopmnearwell.ml import ensemble
from pyopmnearwell.utils import units

dirname: pathlib.Path = pathlib.Path(__file__).parent

# TODO: Generalize this for different stencils.
FEATURE_TO_INDEX: dict[str, int] = {
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
    "time_days": 11,
    "PI_analytical": 12,
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
    RADIUS_IDX = 2
    FGIT_IDX   = 3
    WGIR_IDX   = 4
    TSLSI_IDX  = 5   # time_since_last_shut_in
    LSID_IDX   = 6   # last_shut_in_duration
    TIME_IDX   = 7
    PI_IDX     = 8

    new_features_lst.append(features[..., RADIUS_IDX])   # radius
    new_features_lst.append(features[..., FGIT_IDX])     # total injected volume
    new_features_lst.append(features[..., WGIR_IDX])     # injection rate
    new_features_lst.append(features[..., TSLSI_IDX])    # time_since_last_shut_in
    new_features_lst.append(features[..., LSID_IDX])     # last_shut_in_duration
    new_features_lst.append(features[..., TIME_IDX])     # time_days

    # --- PI feature ---
    PI = features[..., PI_IDX]
    eps = 1e-12

    # Always append PI as a feature (log if WI_log, same convention as before)
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

    ensemble.store_dataset(
        X.astype(np.float32),
        y_out[..., None].astype(np.float32),
        new_data_dirname,
    )

