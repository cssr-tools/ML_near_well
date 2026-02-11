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
    "PI_analytical": 9,
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
    7. PERMEABILITY - upper neighbor
    8. PERMEABILITY - cell
    9. PERMEABILITY - lower neighbor
    10. radius
    11. total injected gas
    12. analytical PI

    Args:
        data_dirname (str | pathlib.Path): _description_
        stencil_size (int, optional): _description_. Defaults to 3.

    """
    # Load data.
    ds: tf.data.Dataset = tf.data.Dataset.load(str(data_dirname))
    features, targets = next(iter(ds.batch(batch_size=len(ds)).as_numpy_iterator()))
    # Add upper and lower cell features to create the training data for the stencil.
    new_features_lst: list[np.ndarray] = []
    for i in range(features.shape[-1] - 4):
        feature: np.ndarray = features[..., i]

        # Pad all local features and scale values.

        # Pressure options:
        if i == 0:
            if trainspecs["pressure_unit"] == "bar":
                feature = feature * units.PASCAL_TO_BAR

            if trainspecs["pressure_padding"] == "zeros":
                padding_mode: str = "constant"
                padding_value: float = 0.0

            # TODO: Fix init padding mode. Where to get the pressure value from? The
            # runspecs only have the ensemble values. The data truncates the init value.
            # elif trainspecs["pressure_padding"] == "init":
            #     padding_mode = "constant"
            #     padding_values  = runspecs_ensemble["constant"]

            elif trainspecs["pressure_padding"] == "neighbor":
                padding_mode = "edge"
                padding_value = 0.0

        # Saturation options:
        elif i == 1:
            if trainspecs["saturation_padding"] == "zeros":
                padding_mode = "constant"
                padding_value = 0.0

        # Permeability options:
        elif i == 2:
            if trainspecs["permeability_log"]:
                feature = np.log10(feature)

            if trainspecs["permeability_padding"] == "zeros":
                padding_mode = "constant"
                padding_value = 0.0

        # Pad the third (layers) feature dimension.
        # TODO: Make this more general
        # Ignore MypY complaining.
        if padding_mode == "constant":
            upper_features = [
                np.pad(  # type: ignore
                    feature[:, :, : -(j + 1), ...],
                    [(0, 0) if k != 2 else ((j + 1), 0) for k in range(feature.ndim)],
                    mode=padding_mode,
                    constant_values=padding_value,
                )
                for j in range(math.floor(stencil_size / 2))
            ]
            lower_features = [
                np.pad(  # type: ignore
                    feature[:, :, (j + 1) :, ...],
                    [(0, 0) if k != 2 else (0, (j + 1)) for k in range(feature.ndim)],
                    mode=padding_mode,
                    constant_values=padding_value,
                )
                for j in range(math.floor(stencil_size / 2))
            ]
        else:
            upper_features = [
                np.pad(  # type: ignore
                    feature[:, :, : -(j + 1), ...],
                    [(0, 0) if k != 2 else ((j + 1), 0) for k in range(feature.ndim)],
                    mode=padding_mode,
                )
                for j in range(math.floor(stencil_size / 2))
            ]
            lower_features = [
                np.pad(  # type: ignore
                    feature[:, :, (j + 1) :, ...],
                    [(0, 0) if k != 2 else (0, (j + 1)) for k in range(feature.ndim)],
                    mode=padding_mode,
                )
                for j in range(math.floor(stencil_size / 2))
            ]

        # Set together stencil.
        new_features_lst.extend(upper_features + [feature] + lower_features)

    # Add back global features.
    # Radius
    new_features_lst.append(features[..., -4])

    # Total injected volume (FGIT)
    new_features_lst.append(features[..., -3])

    # Injection rate (WGIR:INJ0)
    new_features_lst.append(features[..., -2])

    # Analytical PI #############################################################

    # Extract analytical PI
    PI = features[..., -1]

    # Small epsilon to avoid log10(0)
    eps = 1e-12

    if trainspecs["WI_log"]:
        # ---- Fix targets BEFORE log10 ----
        # Clamp WI values: keep WI=0 but shift to eps to avoid log10(-inf)
        targets = np.where(targets <= 0, eps, targets)
        targets = np.log10(targets)

        # ---- Fix PI feature BEFORE log10 ----
        PI = np.where(PI <= 0, eps, PI)
        new_features_lst.append(np.log10(PI))
    else:
        # No log transform → just ensure PI has no NaN
        PI = np.nan_to_num(PI, nan=0.0, posinf=0.0, neginf=0.0)
        new_features_lst.append(PI)

    # Analytical WI is not needed for training.

    # Build final feature tensor
    new_features: np.ndarray = np.stack(new_features_lst, axis=-1)

    # Select chosen features
    new_features = new_features[
        ..., [FEATURE_TO_INDEX[feature] for feature in trainspecs["features"]]
    ]

    # Flatten dataset and store
    ensemble.store_dataset(
        new_features.reshape(-1, new_features.shape[-1]),
        targets.flatten()[..., None],
        new_data_dirname,
    )
