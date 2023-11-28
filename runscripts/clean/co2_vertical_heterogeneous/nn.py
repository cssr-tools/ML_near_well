import math
import os
import pathlib
from typing import Any, Literal, Optional

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pyopmnearwell.ml import ensemble, nn
from pyopmnearwell.utils import plotting, units
from runspecs import runspecs_ensemble, trainspecs
from tensorflow import keras

dirname: str = os.path.dirname(__file__)

# TODO: Generalize this for different stencils.
FEATURE_TO_INDEX: dict[str, int] = {
    "pressure_upper": 0,
    "pressure": 1,
    "pressure_lower": 2,
    "saturation_upper": 3,
    "saturation": 4,
    "saturation_lower": 5,
    "permeability_upper": 6,
    "permeability": 7,
    "permeability_lower": 8,
    "radius": 9,
    "total_injected_volume": 10,
    "PI_analytical": 11,
}

plotted_values_units: dict[str, str] = {"WI": r"[m^4 \cdot s/kg]", "bhp": "[Pa]"}
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
    10. total injected gas
    11. analytical WI

    Args:
        data_dirname (str | pathlib.Path): _description_
        stencil_size (int, optional): _description_. Defaults to 3.

    """
    # Restructure data.
    ds: tf.data.Dataset = tf.data.Dataset.load(str(data_dirname))
    features, targets = next(iter(ds.batch(batch_size=len(ds)).as_numpy_iterator()))

    # Add uppwer and lower cell features to create the training data for the stencil.
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
                    [(0, 0) if i != 2 else ((j + 1), 0) for i in range(feature.ndim)],
                    mode=padding_mode,
                    constant_values=padding_value,
                )
                for j in range(math.floor(stencil_size / 2))
            ]
            lower_features = [
                np.pad(  # type: ignore
                    feature[:, :, (j + 1) :, ...],
                    [(0, 0) if i != 2 else (0, (j + 1)) for i in range(feature.ndim)],
                    mode=padding_mode,
                    constant_values=padding_value,
                )
                for j in range(math.floor(stencil_size / 2))
            ]
        else:
            upper_features = [
                np.pad(  # type: ignore
                    feature[:, :, : -(j + 1), ...],
                    [(0, 0) if i != 2 else ((j + 1), 0) for i in range(feature.ndim)],
                    mode=padding_mode,
                )
                for j in range(math.floor(stencil_size / 2))
            ]
            lower_features = [
                np.pad(  # type: ignore
                    feature[:, :, (j + 1) :, ...],
                    [(0, 0) if i != 2 else (0, (j + 1)) for i in range(feature.ndim)],
                    mode=padding_mode,
                )
                for j in range(math.floor(stencil_size / 2))
            ]

        # Set together stencil.
        new_features_lst.extend(upper_features + [feature] + lower_features)

    # Add back global features.
    # Radius
    new_features_lst.append(features[..., -3])

    # Total injected volume
    new_features_lst.append(features[..., -2])

    # Analytical PI
    if trainspecs["WI_log"]:
        targets = np.log10(targets)
        new_features_lst.append(np.log10(features[..., -1]))
    else:
        new_features_lst.append(features[..., -1])

    # Analytical WI is not needed for training.

    new_features: np.ndarray = np.stack(new_features_lst, axis=-1)

    # Select the correct features from the train specs
    new_features: np.ndarray = new_features[
        ..., [FEATURE_TO_INDEX[feature] for feature in trainspecs["features"]]
    ]

    # Flatten the dataset and store it
    ensemble.store_dataset(
        new_features.reshape(-1, new_features.shape[-1]),
        targets.flatten()[..., None],
        new_data_dirname,
    )


def tune_and_train(
    trainspecs: dict[str, Any],
    data_dirname: str | pathlib.Path,
    nn_dirname: str | pathlib.Path,
):
    train_data, val_data, test_data = nn.scale_and_prepare_dataset(
        data_dirname,
        feature_names=trainspecs["features"],
        savepath=nn_dirname,
        scale=trainspecs["MinMax_scaling"],
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        shuffle="first",
    )

    train_features, train_targets = train_data
    assert not np.any(np.isnan(train_features))
    assert not np.any(np.isnan(train_targets))
    val_features, val_targets = val_data
    assert not np.any(np.isnan(train_features))
    assert not np.any(np.isnan(val_targets))
    test_features, test_targets = test_data
    assert not np.any(np.isnan(train_features))
    assert not np.any(np.isnan(val_targets))

    # # Adapt the layers when using z-normalization.
    # TODO: Implement this in ```nn.tune`` somehow.
    # if trainspecs["Z-normalization"]:
    #     model.layers[0].adapt(train_data[0])
    #     model.layers[-1].adapt(train_data[1])

    # Add sample weight equal to abs. target value to get percentage loss.
    if trainspecs.get("percentage_loss", False):
        sample_weight: np.ndarray = 1 / (np.abs(train_targets) + np.finfo(float).eps)
    else:
        sample_weight = np.ones_like(train_targets)

    # Get model by hyperparameter tuning.
    model, tuner = nn.tune(
        len(trainspecs["features"]),
        1,
        train_data,
        val_data,
        objective=trainspecs.get("tune_objective", "val_loss"),
        sample_weight=sample_weight,
    )
    nn.save_tune_results(tuner, nn_dirname)
    nn.train(
        model,
        train_data,
        val_data,
        nn_dirname,
        recompile_model=False,
        sample_weight=sample_weight,
        kerasify=trainspecs["kerasify"],
    )


def reload_data(
    runspecs: dict[str, Any],
    trainspecs: dict[str, Any],
    data_dirname: str | pathlib.Path,
    step_size_x: int = 1,
    step_size_t: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    # Get feature shape. Along the corresponding dimensions ``num_timesteps`` and
    # ``num_xcells`` were reduced, by feature[:, ::step_size_t, ::, ::step_size_x]. We
    # calculate the adjusted dimensions with ``math.ceil``.
    num_timesteps: int = math.ceil(
        (runspecs["constants"]["INJECTION_TIME"] * 10) / step_size_t
    )
    num_layers: int = runspecs["constants"]["NUM_LAYERS"]
    num_xcells: int = math.ceil((runspecs["constants"]["NUM_XCELLS"] - 1) / step_size_x)
    num_features: int = len(trainspecs["features"])

    # Load flattened data and reshape.
    ds: tf.data.Dataset = tf.data.Dataset.load(str(data_dirname))
    features, targets = next(iter(ds.batch(batch_size=len(ds)).as_numpy_iterator()))
    features = features.reshape(-1, num_timesteps, num_layers, num_xcells, num_features)
    targets = targets.reshape(-1, num_timesteps, num_layers, num_xcells, 1)
    return features, targets


def plot_member(
    features: np.ndarray,
    WI_data: np.ndarray,
    member: int,
    model: keras.Model,
    nn_dirname: str | pathlib.Path,
    savepath: str | pathlib.Path,
    pressure_index: Optional[int] = None,
    inj_rate_index: Optional[int] = None,
    plotted_value: Literal["bhp", "WI"] = "WI",
    x_axis: Literal["time", "radius"] = "radius",
    comparison: Literal["timesteps", "layers"] = "layers",
    fixed_index: int = 0,
) -> None:
    nn_dirname = pathlib.Path(nn_dirname)

    # # Comparison nn WI vs. Peaceman WI vs. data WI for 3 layers for the first ensemble
    # # member.
    # timesteps: np.ndarray = np.linspace(0, 1, features.shape[-3]) / 1  # unit: [day]
    if x_axis == "time" and comparison == "timesteps":
        raise ValueError("x_axis and comparison cannot both be time")

    if plotted_value == "bhp":
        # For "bhp" to be plotted from the neural network output, the pressure and injection
        # rate need to be known.
        if pressure_index is None or inj_rate_index is None:
            raise ValueError(
                f"plotted_value = {plotted_value} requires values for pressure_index and inj_rate_index"
            )
        # For "bhp" data to be plotted it has to be included in data.
        else:
            # TODO: Fix plotting of bhp values. Either the bhp needs to be passed
            # themselves or pressure values must include the well cell.
            raise ValueError("bhp values must be part of the features")

    # Get target WI and feature values.
    feature_member: np.ndarray = features[member]
    WI_data_member: np.ndarray = WI_data[member]

    if plotted_value == "bhp":
        # Reconstruct the bhp from data WI, analytical WI, NN WI.
        pressure_member: np.ndarray = features[member, ..., pressure_index]
        inj_rate_member: np.ndarray = features[member, ..., inj_rate_index]
        bhp_reconstructed_member = inj_rate_member / WI_data_member + pressure_member

        data_member: np.ndarray = features[member, ..., 0, pressure_index]
        analytical_member = inj_rate_member / analytical_member + pressure_member
    elif plotted_value == "WI":
        pass

    # The comparison axis comes first, the fixed axis (i.e., one point gets selected
    # with ``fixed_index``) comes second, and the plotted axis comes third. We swap axes
    # s.t. this is correct.
    if comparison == "layers":
        comp_axis = range(features.shape[-3])
        # Swap time step and layer axes.
        feature_member = np.swapaxes(feature_member, 0, 1)
        WI_data_member = np.swapaxes(WI_data_member, 0, 1)
        # analytical_member = np.swapaxes(analytical_member, 0, 1)
        if plotted_value == "bhp":
            bhp_reconstructed_member = np.swapaxes(bhp_reconstructed_member, 0, 1)
    elif comparison == "timesteps":
        comp_axis = range(features.shape[1])

    if x_axis == "time":
        x_values: np.ndarray = np.arange(features.shape[1])
        # In this case, comparison must be "layers", hence timesteps are on the second
        # axis. We swap again with radius
        feature_member = np.swapaxes(feature_member, 1, 2)
        WI_data_member = np.swapaxes(WI_data_member, 1, 2)
        # analytical_member = np.swapaxes(analytical_member, 1, 2)
        if plotted_value == "bhp":
            bhp_reconstructed_member = np.swapaxes(bhp_reconstructed_member, 1, 2)
    elif x_axis == "radius":
        x_values = np.arange(features.shape[-2])

    fig: Figure = plt.figure()

    # Map for all given comparisons.
    for num_comp, color in zip(
        comp_axis, plt.cm.rainbow(np.linspace(0, 1, len(comp_axis)))
    ):
        input: np.ndarray = feature_member[num_comp, fixed_index]
        WI_nn: np.ndarray = nn.scale_and_evaluate(
            model, input, nn_dirname / "scalings.csv"
        )

        if trainspecs["WI_log"]:
            # WI_analytical = 10**WI_analytical
            WI_data_member = 10**WI_data_member
            WI_nn = 10**WI_nn

        plt.plot(
            x_values,
            WI_nn,
            label=rf"{comparison} {num_comp}: NN ${plotted_value}$",
            color=color,
        )
        plt.scatter(
            x_values,
            WI_data_member[num_comp, fixed_index],
            label=rf"{comparison} {num_comp}: data ${plotted_value}$",
            color=color,
        )
    plt.legend()
    plt.xlabel(rf"${x_axis}\, {x_axis_units[x_axis]}$")
    plt.ylabel(rf"${plotted_value}\, {plotted_values_units[plotted_value]}$")
    plt.title(
        rf"${plotted_value}$ plotted vs {x_axis} for various {comparison}"
        + f" at {comparisons_inverse[comparison]} {fixed_index}"
    )
    plotting.save_fig_and_data(
        fig,
        savepath,
    )


if __name__ == "__main__":
    data_dirname: pathlib.Path = (
        pathlib.Path(dirname) / f"dataset_{runspecs_ensemble['name']}"
    )

    # Directory for the new data.
    new_data_dirname: pathlib.Path = (
        pathlib.Path(dirname)
        / f"dataset_{runspecs_ensemble['name']}_{trainspecs['name']}"
    )
    nn_dirname: pathlib.Path = (
        pathlib.Path(dirname) / f"nn_{runspecs_ensemble['name']}_{trainspecs['name']}"
    )

    # new_data_dirname.mkdir(exist_ok=True)
    # nn_dirname.mkdir(exist_ok=True)

    # restructure_data(data_dirname, new_data_dirname, trainspecs)
    # tune_and_train(trainspecs, new_data_dirname, nn_dirname)

    model: keras.Model = keras.models.load_model(nn_dirname / "bestmodel.keras")
    features, targets = reload_data(
        runspecs_ensemble, trainspecs, new_data_dirname, step_size_x=5, step_size_t=5
    )
    for i in range(0, 49, 10):
        plot_member(
            features,
            targets,
            i,
            model,
            nn_dirname,
            nn_dirname / f"WI_vs_radius_member_{i}",
            fixed_index=4,
        )
        plot_member(
            features,
            targets,
            i,
            model,
            nn_dirname,
            nn_dirname / f"WI_vs_time_member_{i}",
            x_axis="time",
        )
