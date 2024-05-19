import pathlib
from contextlib import nullcontext as does_not_raise
from typing import Any, Dict, Literal, Optional
from unittest import mock

import numpy as np
import pytest
from matplotlib.figure import Figure
from tensorflow import keras

# Import the function to be tested
from .utils import plot_member, read_and_plot_bhp

# Define test data
features = np.random.rand(10, 5, 20, 3)
WI_data = np.random.rand(10, 5, 20)
member = 0
savepath = "/path/to/save"
trainspecs = {"architecture": "fcnn"}
model = keras.Sequential()
nn_dirname = "/path/to/nn"
y_param = "WI"
x_param = "radius"
comparison_param = "layers"
fixed_param_index = 0
NUM_INPUTS: int = 3


# @pytest.fixture(params=[])
# def setup_network(
#     tmp_path, request
# ) -> tuple[keras.Model, pathlib.Path, dict[str, Any]]:
#     model: keras.Model = keras.Sequential([keras.Input(shape=(NUM_INPUTS,))])
#     nn_dirname: pathlib.Path = tmp_path
#     # Create scaling file
#     with (nn_dirname / "scalings.csv").open("w") as f:
#         f.writelines([])
#     trainspecs: dict[str, Any] = {"architecture": "fcnn"}
#     return model, nn_dirname, trainspecs


# # Define test cases
# @pytest.mark.parametrize("features", [(np.random.rand(10, 5, 20, NUM_INPUTS))])
# @pytest.mark.parametrize("WI_data", [(np.random.rand(10, 5, 20))])
# @pytest.mark.parametrize("member, savepath,, fixed_param_index", [(0, tmp_path, 0)])
# @pytest.mark.parametrize("y_param", [("WI"), ("WI_log"), ("p_w")])
# @pytest.mark.parametrize("x_param", [("time"), ("radius")])
# @pytest.mark.parametrize("comparison_param", [("time"), ("layer")])
# @pytest.mark.parametrize(
#     "kwargs, expected_error",
#     [
#         ({}, ValueError),  # Missing required kwargs
#         ({"pressure_index": 1}, pytest.raises(ValueError)),  # Missing required kwargs
#         ({"inj_rate_index": 2}, pytest.raises(ValueError)),  # Missing required kwargs
#         ({"radius_index": 3}, pytest.raises(ValueError)),  # Missing required kwargs
#         ({"final_time": 10.0}, pytest.raises(ValueError)),  # Missing required kwargs
#         ({"WI_analytical_index": 4}, does_not_raise()),  # All required kwargs provided
#     ],
# )
# def test_plot_member(
#     features,
#     WI_data,
#     member,
#     savepath,
#     nn_dirname,
#     y_param,
#     x_param,
#     comparison_param,
#     kwargs: Dict[str, Any],
#     expected_exception,
# ):
#     with expected_exception:
#         plot_member(
#             features,
#             WI_data,
#             member,
#             savepath,
#             trainspecs,
#             model,
#             nn_dirname,
#             y_param,
#             x_param,
#             comparison_param,
#             fixed_param_index,
#             **kwargs,
#         )


class MockEclSum:
    # Mocking the EclSum class and its methods
    def __init__(self, file_path):
        pass

    def get_values(self, key, report_only):
        if key == "WBHP:INJ0":
            return np.random.rand(10)
        elif key == "TIME":
            return np.linspace(0, 1, 10)


# @mock.patch("ecl.EclSum", MockEclSum)
@mock.patch("ecl.summary.ecl_sum.EclSum", MockEclSum)
@pytest.mark.parametrize(
    ["sum_files", "labels", "colors", "linestyles"],
    [
        (
            [
                pathlib.Path("path") / " sum_file1",
                pathlib.Path("path") / " sum_file2",
                pathlib.Path("path") / " sum_file3",
            ],
            ["Label 1", "Label 2", "Label 3"],
            ["red", "blue", "green"],
            ["solid", "dashed", "dotted"],
        )
    ],
)
def test_read_and_plot_bhp(
    tmp_path,
    sum_files: list[pathlib.Path],
    labels: list[str],
    colors: list[str],
    linestyles: list[str],
):
    savepath: pathlib.Path = tmp_path / "read_and_plot.svg"
    read_and_plot_bhp(sum_files, labels, colors, linestyles, savepath)
    assert (savepath).exists()
