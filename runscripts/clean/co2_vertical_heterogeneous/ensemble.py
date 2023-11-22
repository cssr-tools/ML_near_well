import math
import os
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from pyopmnearwell.ml import ensemble
from pyopmnearwell.ml.data import BaseDataset
from pyopmnearwell.utils import formulas, plotting, units
from runspecs import runspecs_ensemble_1 as runspecs_ensemble

dirname: pathlib.Path = pathlib.Path(__file__).parent

ensemble_dirname: pathlib.Path = dirname / runspecs_ensemble["name"]
data_dirname: pathlib.Path = dirname / f"dataset_{runspecs_ensemble['name']}"

ensemble_dirname.mkdir(exist_ok=True)
data_dirname.mkdir(exist_ok=True)


def ensemble_and_data(runspecs: dict[str, Any], dirname: str | pathlib.Path) -> None:
    """_summary_

    _extended_summary_

    Args:
        runspecs (dict[str, Any]): _description_
        dirname (str | pathlib.Path): _description_

    """
    co2_ensemble = ensemble.create_ensemble(
        runspecs_ensemble,
        efficient_sampling=[
            f"PERM_{i}" for i in range(runspecs_ensemble["constants"]["NUM_ZCELLS"])
        ]
        + ["INIT_PRESSURE"],
    )
    ensemble.setup_ensemble(
        ensemble_dirname,
        co2_ensemble,
        os.path.join(dirname, "ensemble.mako"),
        recalc_grid=False,
        recalc_sections=True,
        recalc_tables=False,
    )
    data: dict[str, Any] = ensemble.run_ensemble(
        runspecs_ensemble["constants"]["FLOW"],
        ensemble_dirname,
        runspecs_ensemble,
        ecl_keywords=["PRESSURE", "SGAS", "FLOGASI+"],
        init_keywords=["PERMX"],
        summary_keywords=["FGIT"],
        num_report_steps=runspecs_ensemble["constants"]["INJECTION_TIME"] * 10,
    )
    features: np.ndarray = np.array(
        ensemble.extract_features(
            data,
            keywords=["PRESSURE", "SGAS", "PERMX", "FLOGASI+", "FGIT"],
            keyword_scalings={"PRESSURE": units.BAR_TO_PASCAL},
        )
    )
    np.save(str(dirname), features)


class Dataset(BaseDataset):
    """Provide methods to create a dataset from ensemble dataset.

    Features are, in the following order:
    1. Pressure - per cell; unit [Pa]
    2. Saturation - per cell; no unit
    3. Permeability - per cell; unit [mD]
    4. Radii - ; unit [m]
    5. Total injected gas; unit [m^3]
    6. Analytical WI - per cell; unit [...]

    The feature array will have shape ``(num_ensemble_runs, num_timesteps/step_size_t,
    num_layers, num_xcells/step_size_x, num_features)``.
    The target array will have shape ``(num_ensemble_runs, num_timesteps/step_size_t,
    num_layers, num_xcells/step_size_x, 1)``

    """

    def __init__(self, features: np.ndarray, num_ensemble_runs: int) -> None:
        """_summary_

        Note: It is always assumed that if reshaped with ... ordering, the features
        array will have the following structure:

        ``shape == (num_ensemble_runs, num_timesteps, num_layers, num_zcells_per_layer,
        num_xcells, features)``

        """
        self.num_members: int = num_ensemble_runs
        self.num_timesteps: int = runspecs_ensemble["constants"]["INJECTION_TIME"] * 10

        self.num_layers: int = runspecs_ensemble["constants"]["NUM_LAYERS"]
        self.num_zcells: int = (
            runspecs_ensemble["constants"]["NUM_ZCELLS"]
            / runspecs_ensemble["constants"]["NUM_LAYERS"]
        )
        self.num_xcells: int = (
            runspecs_ensemble["constants"]["NUM_ZCELLS"]
            / runspecs_ensemble["constants"]["NUM_LAYERS"]
        )

        self.single_feature_shape: tuple[int, int, int, int] = (
            self.num_members,
            self.num_timesteps,
            self.num_layers,
            self.num_xcells,
        )
        self.features: np.ndarray = features.reshape()
        # NOTE: Each data array is reshaped into ``(num_members, num_report_steps, num_layers,
        # num_zcells/num_layers, num_xcells).
        #   - The second to last axis (i.e., all vertical cells in one layer) is eliminated by
        #     averaging or summation.
        #   - The last axis (i.e., all horizontal cells in one layer) is eliminated by
        #     integration or picking a single value.

    def create_ds(
        self, step_size_x: int = 1, step_size_t: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create a dataset by collecting and processing various features and target data.

        Args:
            step_size_x (int): Step size for reducing the size of data along the x-axis
                per layer.
            step_size_t (int): Step size for reducing the size of data along the
                timesteps.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the processed feature data
            and the target data.

        """
        feature_lst: list[np.ndarray] = []

        # Get all data.
        feature_lst.append(self.get_pressures(self.features))
        cell_center_radii, cell_boundary_radii = self.get_radii(radii_file)
        feature_lst.append(cell_center_radii)
        feature_lst.append(self.get_saturations(self.features, cell_boundary_radii))
        feature_lst.append(self.get_permeabilities(self.features))
        feature_lst.append(self.get_tot_injected_gas(self.features))
        feature_lst.append(
            self.get_analytical_WI(feature_lst[0], feature_lst[1], feature_lst[2])
        )
        WI_data: np.ndarray = self.get_data_WI(self.features, feature_lst[0])

        # Reduce size of data and unify shape.
        feature_shape: tuple = np.broadcast_shapes(
            *[feature.shape for feature in feature_lst]
        )
        feature_lst[:] = [
            np.broadcast_to(feature, feature_shape) for feature in feature_lst
        ]
        feature_lst[:] = [
            self.reduce_data_size(feature, step_size_x, step_size_t)
            for feature in feature_lst
        ]

        WI_data = self.reduce_data_size(WI_data, step_size_x, step_size_t)

        return np.stack(feature_lst, axis=-1), WI_data


if __name__ == "__main__":
    pass
