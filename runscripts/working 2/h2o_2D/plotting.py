import pathlib

import ecl
import numpy as np
from matplotlib import pyplot as plt
from pyopmnearwell.utils import plotting


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
    for i in range(len(sum_files)):
        summary_file: ecl.summary.ecl_sum.EclSum = ecl.summary.ecl_sum.EclSum(
            str(sum_files[i])
        )
        bhp_values: np.ndarray = np.array(
            summary_file.get_values("WBHP:INJ0", report_only=True)
        )
        time: np.ndarray = np.array(summary_file.get_values("TIME", report_only=True))
        ax.plot(
            time,
            bhp_values,
            label=labels[i],
            color=colors[i],
            linestyle=linestyles[i],
        )
    # Fix legend and axes titles.
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_xlabel("Time since injection start (days)")
    ax.set_ylabel("Bottom hole pressure (bar)")

    plotting.save_fig_and_data(fig, savepath)


dirname: pathlib.Path = pathlib.Path(__name__)

summary_files: list[pathlib.Path] = [dirname / str(i) for i in range(5)]
read_and_plot_bhp()
