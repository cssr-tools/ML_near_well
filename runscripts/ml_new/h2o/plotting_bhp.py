import pathlib

import ecl
import numpy as np
import seaborn as sns
from ecl.summary.ecl_sum import EclSum
from matplotlib import pyplot as plt
from pyopmnearwell.utils import plotting

dirname: pathlib.Path = pathlib.Path(__file__).parent

# Set seaborn style.
sns.set_theme(context="paper", style="whitegrid")


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

    ax.set_yscale("log")
    plotting.save_fig_and_data(fig, savepath)


summary_files: list[pathlib.Path] = [
    dirname / "integration" / "run_2" / "output" / "25X25M_PEACEMAN.SMSPEC",
    dirname / "integration" / "run_0" / "output" / "125X125M_NN.SMSPEC",
    dirname / "integration" / "run_1" / "output" / "125X125M_PEACEMAN.SMSPEC",
]
labels: list[str] = [
    "Fine-scale benchmark",
    "125x125m NN",
    "125x125m Peaceman",
]
colors: list[str] = (
    ["black"]
    + list(plt.cm.Blues(np.linspace(0.7, 0.3, 1)))
    + list(plt.cm.Greys(np.linspace(0.7, 0.3, 1)))
)
linestyles: list[str] = [
    "solid",
    "dashed",
    "dotted",
]
read_and_plot_bhp(summary_files, labels, colors, linestyles, dirname / "bhp.svg")
