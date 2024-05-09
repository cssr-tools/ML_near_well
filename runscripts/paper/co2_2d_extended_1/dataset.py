import os
import pathlib
from typing import Any

import numpy as np
from pyopmnearwell.ml import ensemble
from pyopmnearwell.ml.data import BaseDataset
from pyopmnearwell.utils import formulas


class CO2_2D_Dataset(BaseDataset):
    """Create a tensorflow dataset from ensemble data for 2D CO2 simulations.

    Features are, in the following order:
    1. Pressure - per cell; unit [Pa]
    2. Permeability - per cell; unit [m^2]
    3. Equivalent well radius - per cell; unit [m]
    4. Height - per cell; unit [m]
    5. Total injected gas - global; unit [m^3]
    (6. Analytical WI - per cell; unit [...] -> This is just for comparison and not used
        during training.)

    Input data from the ensemble is assumed to be, in the following order:
    1. PRESSURE - per cell; unit [Pa]
    2. FLOGASI+ - per cell; unit [m/s] ??
    3. PERMX - per cell; unit [m^2]
    4. DZ - per cell; unit [m]
    5. FGIT - global; unit [m^3]


    The feature array will have shape ``(num_ensemble_runs, num_timesteps/step_size_t,
    num_xcells/step_size_x, num_features)``.
    The target array will have shape ``(num_ensemble_runs, num_timesteps/step_size_t,
    num_xcells/step_size_x, 1)``

    """

    def __init__(self, data: np.ndarray, runspecs: dict[str, Any], num_features: int):
        """_summary_

        Note: It is always assumed that if reshaped with ... ordering, the features
        array will have the following structure:

        ``shape == (num_ensemble_runs, num_timesteps, num_layers, num_zcells_per_layer,
        num_xcells, features)``

        """
        # Find out the actual number of ensemble members that ran until the end.
        self.num_members: int = data.shape[0]

        # Compute number of report steps.
        self.num_timesteps: int = int(
            runspecs["constants"]["INJECTION_TIME"]
            * 1
            / runspecs["constants"]["REPORTSTEP_LENGTH"]
        )

        # The well cell and the pore volume cell get disregarded in the final dataset.
        self.num_xcells: int = runspecs["constants"]["NUM_XCELLS"] - 2
        self.num_zcells: int = 1
        self.num_layers: int = 1

        self.data: np.ndarray = data.reshape(
            -1,
            self.num_timesteps,
            self.num_layers,
            self.num_zcells,
            self.num_xcells + 2,
            num_features,
        )
        # Disregard the pore volume cells. The well cells get disregarded later.
        self.data = self.data[..., :-1, :]

        # Find out the actual number of ensemble members that ran until the end.
        self.num_members: int = self.data.shape[0]
        self.single_feature_shape: tuple[int, int, int, int] = (
            self.num_members,
            self.num_timesteps,
            self.num_layers,
            self.num_xcells,
        )

        self.runspecs: dict[str, Any] = runspecs

    def create_ds(
        self,
        ensemble_dirname: pathlib.Path,
        step_size_x: int = 1,
        step_size_t: int = 1,
        calc_analytical_WI: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create a dataset by collecting and processing various features and target data.

        Args:
            ensemble_dirname (pathlib.Path): The directory path where the ensemble
                data is stored. Needed to access ``.GRID`` file.
            step_size_x (int, optional): Step size for reducing the size of data
                along the x-axis. Defaults to 1.
            step_size_t (int, optional): Step size for reducing the size of data
                along the time axis. Defaults to 1.
            calc_analytical_WI (bool, optional): Flag indicating whether to
                calculate analytical WI. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the processed feature data
                and the target data.

        Raises:
            FileNotFoundError: If the preprocessing file is not found.

        """
        feature_lst: list[np.ndarray] = []

        # Get all data.
        # Get pressures.
        feature_lst.append(self.get_vertically_averaged_values(self.data, 0))
        # Get permeabilities.
        feature_lst.append(self.get_homogeneous_values(self.data, 2))
        # Get cell heights.
        feature_lst.append(self.get_homogeneous_values(self.data, 3))
        # Get equivalent radii from preprocessing file.
        cell_center_radii, cell_boundary_radii = self.get_radii(
            ensemble_dirname / "runfiles_0" / "preprocessing" / "GRID.INC"
        )
        # Apply correction from triangle to radial grid.
        cell_center_radii *= formulas.pyopmnearwell_correction()
        cell_boundary_radii *= formulas.pyopmnearwell_correction()
        feature_lst.append(cell_center_radii)
        # Get total injected gas.
        feature_lst.append(self.get_homogeneous_values(self.data, 4))

        # Get data based WI as target.
        WI_data: np.ndarray = self.get_data_WI(
            self.data,
            0,
            1,
        )

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

        # Add analytical WI only now to save computing time. Otherwise it's near
        # impossible, because PVT needs to be called for each datapoint individually.
        if calc_analytical_WI:
            feature_lst.append(
                self.get_analytical_WI(
                    feature_lst[0],
                    np.ones_like(feature_lst[0]),
                    feature_lst[1],
                    self.runspecs["constants"]["INIT_TEMPERATURE"],
                    self.runspecs["constants"]["SURFACE_DENSITY"],
                    radii=feature_lst[3],
                    well_radius=cell_boundary_radii[0],
                    OPM=self.runspecs["constants"]["OPM"],
                )
            )

        return (np.stack(feature_lst, axis=-1), WI_data)
