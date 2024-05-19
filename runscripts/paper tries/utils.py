"""Provide some utility based on the pyopmnearwell module.

Runs on dictionaries that provide run parameters for ensemble simulations, training and
integration.


"""

import inspect
import math
import pathlib
from typing import Any, Literal, Optional

import numpy as np
import tensorflow as tf
from ecl.summary.ecl_sum import EclSum
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pyopmnearwell.ml import ensemble, nn
from pyopmnearwell.utils import plotting, units
from tensorflow import keras

Y_AXIS_UNITS: dict[str, str] = {
    "WI": r"[m^4 \cdot s/kg]",
    "bhp": "[Pa]",
    "perm": "[mD]",
}

X_AXIS_UNITS: dict[str, str] = {"time": "[d]", "radius": "[m]"}

COMP_INVERSE: dict[str, dict[str, str]] = {
    "time": {"radius": "layer"},
    "layer": {"radius": "time step", "time": "radius"},
}

FEATURE_TO_INDEX: dict[str, int] = {}


def full_ensemble(
    runspecs: dict[str, Any],
    ensemble_dirname: str | pathlib.Path,
    ecl_keywords: Optional[list[str]] = None,
    init_keywords: Optional[list[str]] = None,
    summary_keywords: Optional[list[str]] = None,
    **kwargs,
) -> np.ndarray:
    """Create, setup and run an ensemble, then extract data.

    The following data is extracted (in this order):
    - pressure
    - gas saturation
    - permeability x
    - permeability z
    - gas flow rate i (-> get values of well cells for ~connection injection rate)
    - total injected gas volume

    Data will have the following units:
    - pressure: [Pa]
    - permeability: [m^2]
    - gas flow rate i: [m^3/s]
    - total injected gas volume: [m^3]

    Args:
        runspecs (dict[str, Any]): Dictionary containing at least the following keys:
            - "variables": dict[str, tuple[float, float, int]] - Each tuple gives the
                min and max value and number of different values for the variable.
            - "constants": dict[str, float] - The union of all keys in "variables" and
                "constants" must contain all variables in the ``ensemble.mako`` file.
            - "OPM": str | pathlib.Path - Path to an OPM installation.
            - "FLOW": str | pathlib.Path - Path to a flow executable.
        ensemble_dirname (str | pathlib.Path): _description_
        ecl_keywords (Optional[list[str]]):
        init_keywords (Optional[list[str]]):
        summary_keywords (Optional[list[str]]):
        **kwargs: Is passed to ``setup_ensemble`` and ``run_ensemble``. Possible
            parameters are:
            - recalc_grid (bool, optional): Whether to recalculate ``GRID.INC`` for each
                ensemble member. Defaults to False.
            - recalc_tables (bool, optional): Whether to recalculate ``TABLES.INC`` for
                each ensemble member. Defaults to False.
            - recalc_sections (bool, optional): Whether to recalculate ``GEOLOGY.INC``
                and ``REGIONS.INC`` for each ensemble member. Defaults to False.
            - step_size_time: Save data only for every ``step_size_time`` report step.
                Default is 1.
            - step_size_cell: Save data only for every ``step_size_cell`` grid cell.
                Default is 1.

    Returns:
        np.ndarray (``shape=(num_report_steps//step_size_time,
        num_grid_cells//step_size_cell,
        len(ecl_keywords + init_keywords + summary_keywords))``):
            Contains the data in the given order.

    """
    if ecl_keywords is None:
        ecl_keywords = []
    if init_keywords is None:
        init_keywords = []
    if summary_keywords is None:
        summary_keywords = []

    ensemble_dirname = pathlib.Path(ensemble_dirname)

    # Use efficient sampling for all variables.
    ensemble_dict = ensemble.create_ensemble(
        runspecs,
        efficient_sampling=runspecs["variables"],
    )

    # It is assumed that the mako is in the parent directory of ``ensemble_dirname``.
    # Making this more flexible requires some more work.
    ensemble.setup_ensemble(
        ensemble_dirname,
        ensemble_dict,
        ensemble_dirname / ".." / "ensemble.mako",
        **kwargs,
    )
    data: dict[str, Any] = ensemble.run_ensemble(
        runspecs["constants"]["FLOW"],
        ensemble_dirname,
        runspecs,
        ecl_keywords=ecl_keywords,
        init_keywords=init_keywords,
        summary_keywords=summary_keywords,
        num_report_steps=runspecs["constants"]["INJECTION_TIME"]
        * runspecs["constants"]["REPORTSTEP_LENGTH"],
        **kwargs,
    )
    features: np.ndarray = np.array(
        ensemble.extract_features(
            data,
            keywords=ecl_keywords + init_keywords + summary_keywords,
            # Pressure outputs are in and [bar], but OPM uses [Pa] internally.
            keyword_scalings=kwargs.get("keyword_scalings", {}),
        )
    )
    return features


def pad_data(
    trainspecs: dict[str, Any],
    data_dirname: str | pathlib.Path,
    new_data_dirname: str | pathlib.Path,
    stencil_size: int = 3,
    **kwargs,
) -> None:
    """Extend data to fit bigger stencils and pad at the domain boundaries.

    Furthermore the data is scaled. The final dataset will be in a flattened shape.

    Note: The local features inside the stencil always (!) range FROM upper TO lower
        cells.

    TODO: Generalize this for different stencils.
    The new features are in the order given by trainspecs:
    1. PRESSURE - upper neighbor
    2. PRESSURE - cell
    3. PRESSURE - lower neighbor
    4. SATURATION - upper neighbor
    5. SATURATION - cell
    6. SATURATION - lower neighbor
    7. PERMEABILITY - upper neighbor
    8. PERMEABILITY - cell
    9. PERMEABILITY - lower neighbor
    10. Injection rate - upper neighbor
    11. Injection - cell
    12. Injection - lower neighbor
    13. radius
    14. total injected gas
    15. analytical PI

    Args:
        trainspecs (dict[str, Any]): Dictionary containing at least the following keys:
            - "features": dict[str, tuple[float, float, int]] - Each tuple gives the
                min and max value and number of different values for the variable.
        data_dirname (str | pathlib.Path): Where to load the original tensorflow dataset
            from.
        new_data_dirname (str | pathlib.Path): Where store the new tensorflow dataset.
        stencil_size (int, optional): Local features are extended to this size. Defaults
            to 3.
        **kwargs: Is passed to ``setup_ensemble`` and ``run_ensemble``. Possible

    """
    # Load original data.
    ds: tf.data.Dataset = tf.data.Dataset.load(str(data_dirname))
    features, targets = next(iter(ds.batch(batch_size=len(ds)).as_numpy_iterator()))

    # Loop through all features and add upper and lower cell features for all local
    # features. This is done to create the training data for the stencil.
    # NOTE: The data is created by a ``Dataset`` class.
    new_features_lst: list[np.ndarray] = []
    for i in range(features.shape[-1] - 3):
        feature: np.ndarray = features[..., i]

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

        # Injection rate: -> Use zero padding.
        elif i == 3:
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
    **kwargs,
) -> None:
    """Tune hyperparameters and train with the best model.

    _extended_summary_

    Note:
        - If ``trainspecs["percentage_loss"] == True``, the relative loss (relative to
          target) is only applied during training but not during tuning. This does not
          seem to be possible with ``keras.Tuner``.

    Args:
        trainspecs (dict[str, Any]): Dictionary containing at least the following keys:
            -
        data_dirname (str | pathlib.Path): _description_
        nn_dirname (str | pathlib.Path): Path to store the tuning results, the trained
            networks and logs.
        **kwargs: Gets (among others) passed to ``nn.train`` and ``nn.tune``. Possible
            parameters are:
            - train_split (float)
            - val_split (float)
            - test_split (float)
            - lr (float)
            - epochs (int)
            - bs (int)
            - patience (int)
            - lr_patience (int)
            - loss_func (str)
            - objective (str)
            - max_trials (int)
            - executions_per_trial (int)


    """
    # Create datasets and check that they are not empty.
    train_data, val_data, test_data = nn.scale_and_prepare_dataset(
        data_dirname,
        feature_names=trainspecs["features"],
        savepath=nn_dirname,
        scale=trainspecs.get("MinMax_scaling", True),
        train_split=kwargs.get("train_split", 0.8),
        val_split=kwargs.get("val_split", 0.1),
        test_split=kwargs.get("test_split", 0.1),
        # Shuffle last s.t. training, val, and test split come from different ensemble
        # runs.
        shuffle="last",
    )
    # TODO: The assert fails for empty splits.
    train_features, train_targets = train_data
    assert not np.any(np.isnan(train_features))
    assert not np.any(np.isnan(train_targets))
    val_features, val_targets = val_data
    assert not np.any(np.isnan(val_features))
    assert not np.any(np.isnan(val_targets))
    test_features, test_targets = test_data
    assert not np.any(np.isnan(test_features))
    assert not np.any(np.isnan(test_targets))

    # # Adapt the layers when using z-normalization.
    # TODO: Implement this in ```nn.tune`` somehow.
    # if trainspecs["Z-normalization"]:
    #     model.layers[0].adapt(train_data[0])
    #     model.layers[-1].adapt(train_data[1])

    # Set ``sample_weight`` equal to abs. target value to get percentage loss.
    # NOTE: Tuner does not use sample weights.
    if trainspecs.get("percentage_loss", False):
        sample_weight: np.ndarray = 1 / (np.abs(train_targets) + np.finfo(float).eps)
    # Otherwise, set ``sample_weight`` to 1 to keep original loss.
    else:
        sample_weight = np.ones_like(train_targets)

    # Tune hyperparameters and get best model.
    # TODO: Change to two different kwargs lists.
    tune_args = list(inspect.signature(nn.tune).parameters)
    tune_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in tune_args}
    model, tuner = nn.tune(
        len(trainspecs["features"]),
        kwargs.get("noutputs", 1),
        train_data,
        val_data,
        nn_dirname,
        sample_weight=sample_weight,
        **tune_dict,
    )
    nn.save_tune_results(tuner, nn_dirname)

    train_args = list(inspect.signature(nn.tune).parameters)
    train_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in train_args}
    nn.train(
        model,
        train_data,
        val_data,
        nn_dirname,
        recompile_model=False,
        sample_weight=sample_weight,
        kerasify=trainspecs["kerasify"],
        **train_dict,
    )


def just_train(
    trainspecs: dict[str, Any],
    data_dirname: str | pathlib.Path,
    nn_dirname: str | pathlib.Path,
    model: keras.Model,
    **kwargs,
) -> None:
    """Train a given model without tuning.

    _extended_summary_

    Args:
        trainspecs (dict[str, Any]): Dictionary containing at least the following keys:
            -
        data_dirname (str | pathlib.Path): _description_
        nn_dirname (str | pathlib.Path): Path to store the trained
            networks and logs.
        **kwargs: Possible parameters are:
            - train_split (float): Default is 0.8.
            - val_split (float): Default is 0.1.
            - test_split (float): Default is 0.1.

    """
    train_data, val_data, test_data = nn.scale_and_prepare_dataset(
        data_dirname,
        feature_names=trainspecs["features"],
        savepath=nn_dirname,
        scale=trainspecs["MinMax_scaling"],
        train_split=kwargs.get("train_split", 0.8),
        val_split=kwargs.get("val_split", 0.1),
        test_split=kwargs.get("test_split", 0.1),
        # Shuffle last s.t. training, val, and test split come from different ensemble
        # runs.
        shuffle="last",
    )

    train_features, train_targets = train_data
    assert not np.any(np.isnan(train_features))
    assert not np.any(np.isnan(train_targets))
    val_features, val_targets = val_data
    assert not np.any(np.isnan(val_features))
    assert not np.any(np.isnan(val_targets))
    test_features, test_targets = test_data
    assert not np.any(np.isnan(test_features))
    assert not np.any(np.isnan(test_targets))

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
    nn.train(
        model,
        train_data,
        val_data,
        nn_dirname,
        recompile_model=True,
        sample_weight=sample_weight,
        kerasify=trainspecs["kerasify"],
        epochs=20,
        lr=0.0001,
    )


def reload_data(
    runspecs: dict[str, Any],
    trainspecs: dict[str, Any],
    data_dirname: str | pathlib.Path,
    step_size_x: int = 1,
    step_size_t: int = 1,
    num_xcells: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a dataset and return in shape s.t. member/time/z-axis/x-axis are distinct
    axes.

    _extended_summary_

    Args:
        runspecs (dict[str, Any]): _description_
        trainspecs (dict[str, Any]): _description_
        data_dirname (str | pathlib.Path): _description_
        step_size_x (int, optional): _description_. Defaults to 1.
        step_size_t (int, optional): _description_. Defaults to 1.
        num_xcells (Optional[int], optional): _description_. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_

    """
    # Get feature shape. Along the corresponding dimensions ``num_timesteps`` and
    # ``num_xcells`` were reduced, by feature[:, ::step_size_t, ::, ::step_size_x]. We
    # calculate the adjusted dimensions with ``math.ceil``.
    num_timesteps: int = math.ceil(
        (runspecs["constants"]["INJECTION_TIME"] * 10) / step_size_t
    )
    num_layers: int = runspecs["constants"]["NUM_LAYERS"]
    num_features: int = len(trainspecs["features"])

    # Calc. ``num_xcells`` if not provided.
    if num_xcells is None:
        num_xcells = math.ceil((runspecs["constants"]["NUM_XCELLS"] - 1) / step_size_x)

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
    savepath: str | pathlib.Path,
    trainspecs: Optional[dict[str, Any]] = None,
    model: Optional[keras.Model] = None,
    nn_dirname: Optional[str | pathlib.Path] = None,
    y_param: Literal["p_w", "WI", "WI_log"] = "WI",
    x_param: Literal["time", "radius"] = "radius",
    comparison_param: Literal["time", "layer"] = "layer",
    fixed_param_index: int = 0,
    **kwargs,
) -> None:
    """Plot a value against time or radius. Plot multiple different ... in one plot.

    ``plotted_value`` (y-axis) is plotted against ``x_parameter`` (x-axis). Values from
    all ``comparison_parameter`` are plotted in the same plot.

    Args:
        features (np.ndarray): The input features.
        WI_data (np.ndarray): Data-based WI, i.e., targets.
        member (int): Which ensemble member to plot.
        savepath (str | pathlib.Path): The path to save the plot.
        trainspecs (Optional[dict[str, Any]], optional): The training specifications.
            Defaults to None.
        model (Optional[keras.Model], optional): The trained model. Defaults to None.
        nn_dirname (Optional[str | pathlib.Path], optional): The directory of the neural
            network model. Defaults to None.
        y_param (Literal["p_w", "WI", "WI_log"], optional): The parameter to plot on the
        y-axis. Defaults to "WI".
        x_param (Literal["time", "radius"], optional): The parameter to plot on the
        x-axis. Defaults to "radius".
        comparison_param (Literal["time", "layer"], optional): The parameter to compare
            on the plot. Defaults to "layer".
        fixed_param_index (int, optional): The index of the fixed parameter. Defaults to
            0.

        **kwargs:
            - pressure_index (int): The index of the pressure. Defaults to None.
            - inj_rate_index (int): The index of the injection rate. Defaults to None.
            - final_time (float): The final time. Defaults to None.
            - radius_index (int): The index of the radius. Defaults to None.
            - permeability_index (int): The index of the permeability. Defaults to None.

    Returns:
        None

    Raises:
        ValueError: If ``plotted_value == "p_w"`` but either ``pressure_index`` or
            ``inj_rate_index`` are not provided as kwargs.
        ValueError: If ``x_param == "radius`` and
            ``trainspecs["architecture"] == "rnn"``
        ValueError: If ``x_param == comparison_param == "time"``.
        ValueError: If ``x_param == "time"`` and ``final_time`` is ``None``.
        ValueError: If ``x_param == "radius"`` and ``radius_index`` is ``None``.
        ValueError: If trainspecs["architecture"] is not supported (i.e., differs from
        "fcnn" or "rnn").

    """
    # Ensure ``nn_dirname`` is a pathlib.Path.
    if nn_dirname is not None:
        nn_dirname = pathlib.Path(nn_dirname)

    # Assure that the combination of parameters is valid.
    if x_param == "time" and comparison_param == "time":
        raise ValueError("x_axis and comparison cannot both be time")

    plot_nn: bool = False
    if not all(v is None for v in [trainspecs, model, nn_dirname]):
        if any(v is None for v in [trainspecs, model, nn_dirname]):
            raise ValueError(
                "trainspecs, model, and nn_dirname must all have"
                + " a value to plot a neural network."
            )
        else:
            if x_param == "radius" and trainspecs["architecture"] == "rnn":  # type: ignore
                raise ValueError("x_axis 'radius' is not implemented for rnns.")
            plot_nn = True

    final_time: Optional[float] = kwargs.get("final_time", None)
    if x_param == "time" and final_time is None:
        raise ValueError(
            "A final_time value must be provided (as kwarg) if x_param is 'time'."
        )

    radius_index: Optional[int] = kwargs.get("radius_index", None)
    if x_param == "radius" and radius_index is None:
        raise ValueError(
            "A radius_index value must be provided (as kwarg) if x_param is 'radius'."
        )

    permeability_index: Optional[int] = kwargs.get("permeability_index", None)

    # To plot well pressure, it needs to be computed manually from grid block pressure,
    # injection rate and well index. This requires some extra steps.
    if y_param == "p_w":
        # To plot well pressure, it needs to be computed manually from grid block
        # pressure, injection rate and well index. Thus, the pressure and injection rate
        # need to be known.
        pressure_index: Optional[int] = kwargs.get("pressure_index", None)
        inj_rate_index: Optional[int] = kwargs.get("inj_rate_index", None)
        if pressure_index is None or inj_rate_index is None:
            raise ValueError(
                "plotted_value = 'p_w' requires values for pressure_index and inj_rate_index"
            )

    # Get single member.
    feature_member: np.ndarray = features[member]
    WI_data_member: np.ndarray = WI_data[member]

    WI_analytical_index = kwargs.get("WI_analytical_index", None)
    if WI_analytical_index is not None:
        WI_analytical_member: np.ndarray = features[member, ..., WI_analytical_index]
    else:
        # Create dummy s.t. it does not need to be checked whether WI_analytical_index
        # is None all the time. During plotting it is checked again, so the dummy is not
        # plotted.
        WI_analytical_member = np.zeros_like(features[member, ..., 0])

    # If a model is available, evaluate for the member.
    # NOTE: It is not efficient to do this at this point, but it makes coding the rest
    # easier. A faster solution would be to evaluate once the subarray corresponding to
    # fixed_param_index is selected.
    if plot_nn:
        # NOTE: Pylance complains None not being subscriptable etc., but since plot_nn is
        # True, we already checked for everything.
        saved_shape: list[int] = list(feature_member.shape)
        # TODO: Does this also work with a tuple? -> Gives problems when concatenating
        # with a list.

        # Squeeze all axes except the inputs and possibly time steps.
        if trainspecs["architecture"] == "fcnn":  # type: ignore
            input: np.ndarray = feature_member.reshape(-1, saved_shape[-1])
        elif trainspecs["architecture"] == "rnn":  # type: ignore
            # This is more intricate. The axes need to be switched forth and back s.t.
            # the time axis is second to last.
            input = feature_member.swapaxes(0, -2)
            input = input.reshape(-1, saved_shape[0], saved_shape[-1])
        else:
            raise ValueError(f"{trainspecs['architecture']} is not supported.")  # type: ignore

        output: np.ndarray = nn.scale_and_evaluate(
            model, input, nn_dirname / "scalings.csv"  # type: ignore
        )

        # Reshape back into original form
        if trainspecs["architecture"] == "rnn":  # type: ignore
            # Switch back time axis and the other axis (which are squeezed to one axis).
            output = output.swapaxes(0, -2)
        WI_nn_member: np.ndarray = np.reshape(output, saved_shape[:-1] + [1])
    else:
        # Create dummy s.t. it does not need to be checked whether plot_nn
        # is False all the time. During plotting it is checked again, so the dummy is not
        # plotted.
        WI_nn_member = np.zeros_like(features[member, ..., 0])

    # The well pressure ("p_w") is not directly available and needs to be reconstructed
    # from cell pressure, injection rate and WI by :math:`p_w = q / WI + p_gb`. This is
    # done for the data, analytical and neural network WI (if available).

    if y_param == "p_w":
        # Reconstruct the p_w from data WI, analytical WI, NN WI.
        # Ignore pylance complaining about the indices being unbound. They were assigned
        # in case ``y_param == "p_w"``.
        # NOTE: We already checked that pressure_index and inj_rate_index cannot be
        # unbound.
        pressure_member: np.ndarray = feature_member[..., pressure_index]  # type: ignore
        inj_rate_member: np.ndarray = feature_member[..., inj_rate_index]  # type: ignore
        y_values_data_member: np.ndarray = (
            inj_rate_member / WI_data_member + pressure_member
        )

        y_values_analytical_member: np.ndarray = (
            inj_rate_member / WI_analytical_member + pressure_member
        )
        y_values_nn_member: np.ndarray = (
            inj_rate_member / WI_nn_member + pressure_member
        )

    elif y_param == "WI":
        y_values_data_member = WI_data_member
        y_values_analytical_member = WI_analytical_member
        y_values_nn_member = WI_nn_member

    elif y_param == "WI_log":
        # Rescale WI if it was log for training.
        y_values_data_member = WI_data_member**10
        y_values_analytical_member = WI_analytical_member**10
        y_values_nn_member = WI_nn_member**10

    # Note that all ``plotted_value_data_...`` have  shape
    # ``(num_timesteps, num_layers, num_xcells, :)``. Depending on the function
    # parameters, this shape is transformed as follows before plotting.

    # 1. The fixed axis (i.e., one point gets selected with ``fixed_index``, this is
    #    either the time or radius axis) comes first.
    # 2. The comparison axis (either layers or time) comes second.
    # 3. The axis we plot against comes third.
    # Axes are swapped s.t. this is always correct.

    # First, we bring the fixed axis to the front.
    if comparison_param == "time":
        comp_axis: int = 0
    elif comparison_param == "layer":
        comp_axis = 1

    if x_param == "time":
        x_axis: int = 0
        # Get the time steps.
        x_values: np.ndarray = np.linspace(0, final_time, feature_member.shape[0])  # type: ignore
    elif x_param == "radius":
        x_axis = 2
        # Get the radiii from the first datapoint
        x_values = feature_member[0, 0, :, radius_index]

    # Use sets to find the axis that is left.
    fixed_axis: int = list({0, 1, 2} - {comp_axis, x_axis})[0]

    # Actually reorder the axes and take the fixed point.
    y_values_data_member = np.moveaxis(
        y_values_data_member, (fixed_axis, comp_axis, x_axis), (0, 1, 2)
    )[fixed_param_index]
    y_values_analytical_member = np.moveaxis(
        y_values_analytical_member, (fixed_axis, comp_axis, x_axis), (0, 1, 2)
    )[fixed_param_index]
    y_values_nn_member = np.moveaxis(
        y_values_nn_member, (fixed_axis, comp_axis, x_axis), (0, 1, 2)
    )[fixed_param_index]

    # This is done to access permeability if needed
    feature_member = np.moveaxis(
        feature_member, (fixed_axis, comp_axis, x_axis), (0, 1, 2)
    )[fixed_param_index]

    fig: Figure = plt.figure()
    ax = plt.subplot()

    for num_comp, color in zip(
        range(y_values_data_member.shape[0]),
        plt.cm.Blues(np.linspace(1, 0.5, y_values_data_member.shape[0])),  # type: ignore
    ):
        # Get permeability of comparison if available and rescale if necessary.
        if permeability_index is not None:
            # Take permeability at zeroth horizontal cell.
            permeability: float = feature_member[num_comp, 0, permeability_index]
            if trainspecs is not None and trainspecs["permeability_log"]:
                permeability = 10**permeability
        else:
            permeability = math.nan

        ax.scatter(
            x_values,
            y_values_data_member[num_comp],
            label=rf"{comparison_param} {num_comp}: ${y_param}$ data, $\mathbf{{k}}: {permeability * units.M2_TO_MILIDARCY:.2f}\, {Y_AXIS_UNITS['perm']}$",
            color=color,
        )
        if plot_nn:
            ax.plot(
                x_values,
                tf.squeeze(y_values_nn_member[num_comp]),
                label=rf"{comparison_param} {num_comp}: ${y_param}$ NN",
                color=color,
            )
        if WI_analytical_index is not None:
            ax.plot(
                x_values,
                y_values_analytical_member[num_comp],
                label=rf"{comparison_param} {num_comp}: $y_param$ analytical",
                color=color,
                linestyle="-",
            )

    # Shrink axis by 20% and put legend outside the plot.
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_xlabel(rf"${x_param}\, {X_AXIS_UNITS[x_param]}$")
    ax.set_ylabel(rf"${y_param}\, {Y_AXIS_UNITS[y_param]}$")
    ax.set_title(
        rf"${y_param}$ plotted vs {x_param} for various {comparison_param}s"
        + f" at {COMP_INVERSE[comparison_param][x_param]} {fixed_param_index}"
    )

    plotting.save_fig_and_data(
        fig,
        savepath,
    )


def set_seed(seed: int = 0) -> None:
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def L2_error(model: keras.Model, features, targets):
    pass


def read_and_plot_bhp(
    sum_files: list[pathlib.Path],
    labels: list[str],
    colors: list[str],
    linestyles: list[str],
    savepath: str | pathlib.Path,
):
    """
    Reads the bottom hole pressure (BHP) data from summary files and plots the BHP
    values over time.

    Args:
        sum_files (list[pathlib.Path]): A list of paths to the summary files.
        labels (list[str]): A list of labels for each file.
        colors (list[str]): A list of colors for each file.
        linestyles (list[str]): A list of linestyles for each file.
        savepath (str | pathlib.Path): The path to save the plot.

    Returns:
        None

    """
    assert len(sum_files) == len(labels) == len(colors) == len(linestyles)
    fig, ax = plt.subplots()
    # Plot bhp for all files.
    for sum_file, label, color, linestyle in zip(sum_files, labels, colors, linestyles):
        summary_file: EclSum = EclSum(str(sum_file))
        bhp_values: np.ndarray = np.array(
            summary_file.get_values("WBHP:INJ0", report_only=True)
        )
        time: np.ndarray = np.array(summary_file.get_values("TIME", report_only=True))

        if label.startswith("Fine-scale"):
            linewidth: float = 1.5
        else:
            linewidth = 3.0
        ax.plot(
            time,
            bhp_values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    # Fix legend and axes titles.
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_xlabel("Time since injection start (days)")
    ax.set_ylabel("Bottom hole pressure (bar)")

    plotting.save_fig_and_data(fig, savepath)
