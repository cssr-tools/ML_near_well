import pathlib
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.figure import Figure
from pyopmnearwell.utils import formulas, plotting, units
from runspecs import runspecs_ensemble_2 as runspecs_ensemble

dirname: pathlib.Path = pathlib.Path(__file__).parent

plotted_values_units: dict[str, str] = {"WI": r"[m^4 \cdot s/kg]", "p_w": "[Pa]"}
x_axis_units: dict[str, str] = {"time": "[d]", "radius": "[m]"}
comparisons_inverse: dict[str, str] = {
    "timesteps": "layer",
    "layers": "timestep or radius",
}


def plot_member(
    features: np.ndarray,
    WI_data: np.ndarray,
    member: int,
    WI_analytical_index: int,
    savepath: str | pathlib.Path,
    pressure_index: Optional[int] = None,
    radius_index: Optional[int] = None,
    inj_rate_index: Optional[int] = None,
    final_time: Optional[float] = None,
    plotted_value: Literal["p_w", "WI"] = "WI",
    x_axis: Literal["time", "radius"] = "radius",
    comparison: Literal["timesteps", "layers"] = "layers",
    fixed_index: int = 0,
) -> None:
    if x_axis == "time" and comparison == "timesteps":
        raise ValueError("x_axis and comparison cannot both be time")

    if x_axis == "radius" and radius_index is None:
        raise ValueError("radius_index cannot be None")

    if x_axis == "time" and final_time is None:
        raise ValueError("final_time cannot be None")

    if plotted_value == "p_w":
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

    # Get data and analytical values
    data_member: np.ndarray = WI_data[member]

    analytical_member: np.ndarray = features[member, ..., WI_analytical_index]

    if plotted_value == "bhp":
        # Additionally to data and analytical, reconstruct bhp from WI data. This should
        # align with the data.
        pressure_member: np.ndarray = features[member, ..., pressure_index]
        inj_rate_member: np.ndarray = features[member, ..., inj_rate_index]
        p_w_reconstructed_member = inj_rate_member / data_member + pressure_member

        data_member: np.ndarray = features[member, ..., 0, pressure_index]
        analytical_member = inj_rate_member / analytical_member + pressure_member
    elif plotted_value == "WI":
        pass

    if comparison == "layers":
        comp_axis = range(features.shape[-3])
        data_member = np.swapaxes(data_member, 0, 1)
        analytical_member = np.swapaxes(analytical_member, 0, 1)
        if plotted_value == "p_w":
            p_w_reconstructed_member = np.swapaxes(p_w_reconstructed_member, 0, 1)
    elif comparison == "timesteps":
        comp_axis = range(features.shape[1])

    if x_axis == "time":
        x_values: np.ndarray = np.arange(features.shape[1])
        x_values: np.ndarray = np.linspace(0, final_time, features.shape[1])
        # In this case, comparison must be "layers", hence timesteps are on the second
        # axis. We swap again with radius
        data_member = np.swapaxes(data_member, 1, 2)
        analytical_member = np.swapaxes(analytical_member, 1, 2)
        if plotted_value == "p_w":
            p_w_reconstructed_member = np.swapaxes(p_w_reconstructed_member, 1, 2)
    elif x_axis == "radius":
        x_values = features[0, 0, 0, :, radius_index]

    fig: Figure = plt.figure()
    ax = plt.subplot()

    # Map for all given comparisons.
    for num_comp, color in zip(
        comp_axis, plt.cm.rainbow(np.linspace(0, 1, len(comp_axis)))
    ):
        # Plot bhp predicted by Peaceman and data vs actual bhp in the upper layer.
        # NOTE: bhp predicted by data and actual bhp should be identical.
        ax.scatter(
            x_values,
            data_member[num_comp, fixed_index],
            label=rf"{comparison} {num_comp}: data ${plotted_value}$",
            color=color,
        )
        ax.plot(
            x_values,
            analytical_member[num_comp, fixed_index],
            label=rf"{comparison} {num_comp}: analytical",
            color=color,
            linestyle="-",
        )

        if plotted_value == "p_w":
            ax.plot(
                x_values,
                p_w_reconstructed_member[num_comp, fixed_index],
                label=rf"Layer {num_comp}: calculated from data-driven $WI$",
                color=color,
                linestyle="--",
            )

    # Shrink axis by 20% and put legend outside the plot.
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_xlabel(rf"${x_axis}\, {x_axis_units[x_axis]}$")
    ax.set_ylabel(rf"${plotted_value}\, {plotted_values_units[plotted_value]}$")
    ax.set_title(
        rf"${plotted_value}$ plotted vs {x_axis} for various {comparison}"
        + f" at {comparisons_inverse[comparison]} {fixed_index}"
    )
    plotting.save_fig_and_data(
        fig,
        savepath,
    )


def main():
    ensemble_dirname: pathlib.Path = dirname / runspecs_ensemble["name"]
    data_dirname: pathlib.Path = dirname / f"dataset_{runspecs_ensemble['name']}"

    ds: tf.data.Dataset = tf.data.Dataset.load(str(data_dirname))
    features, targets = next(iter(ds.batch(batch_size=len(ds)).as_numpy_iterator()))

    for i in range(0, 49, 10):
        plot_member(
            features,
            targets,
            i,
            5,
            ensemble_dirname / f"WI_vs_radius_member_{i}_timestep{4}",
            fixed_index=4,
            radius_index=3,
        )
        plot_member(
            features,
            targets,
            i,
            5,
            ensemble_dirname / f"WI_vs_radius_member_{i}_timestep{0}",
            fixed_index=0,
            radius_index=3,
        )
        plot_member(
            features,
            targets,
            i,
            5,
            ensemble_dirname / f"WI_vs_time_member_{i}",
            x_axis="time",
            final_time=5,
        )
        # plot_member(
        #     features,
        #     targets,
        #     i,
        #     5,
        #     ensemble_dirname / f"bhp_vs_radius_member_{i}",
        #     pressure_index=0,
        #     inj_rate_index=5,
        #     plotted_value="bhp",
        # )


if __name__ == "__main__":
    main()
