from __future__ import annotations

import json
import math
import pathlib
from typing import Any

import numpy as np
from pyopmnearwell.ml.upscale import BaseUpscaler


class CO2_2D_Upscaler(BaseUpscaler):
    """Upscale ensemble data from 2D CO2 simulations to coarse cartesian cells and
    arange into ``np.ndarrays`` to be used in a ``tensorflow`` dataset.

    Features are, in the following order:
    1. Pressure - per cell; unit [Pa]
    2. Geometrical part of WI; unit [~]
    3. Total injected gas - global; unit [m^3]
    (4. Analytical WI - per cell; unit [...] -> This is just for comparison and not used
        during training.)

    Input data from the ensemble is assumed to be, in the following order:
    1. PRESSURE - per cell; unit [Pa]
    2. FLOGASI+ - per cell; unit [m^3/s] ??
    3. PERMX - per cell; unit [m^2]
    4. DZ - per cell; unit [m]
    5. FGIT - global; unit [m^3]


    The feature array will have shape ``(num_ensemble_runs, num_timesteps/step_size_t,
    num_xcells/step_size_x, num_features)``.
    The target array will have shape ``(num_ensemble_runs, num_timesteps/step_size_t,
    num_xcells/step_size_x, 1)``

    """

    def __init__(
        self,
        data: np.ndarray,
        runspecs: dict[str, Any],
        data_dim: int = 5,
        angle: float = math.pi / 3,
    ):
        """_summary_

        Note: It is always assumed that if reshaped with ... ordering, the features
        array will have the following structure:

        ``shape == (num_ensemble_runs, num_timesteps, num_layers, num_zcells_per_layer,
        num_xcells, features)``

        Args:
            data (np.ndarray): Data from the ensemble simulation. See above.
            runspecs (dict[str, Any]): The runspecs dictionary from the ensemble.
            data_dim (int): Dimension of a single data point. Should not be changed
                unless for very good reason. Default is 5.
            angle (float): Angle between both sides of the triangle grid.
                Default is ``math.pi / 3``.

        """
        # Compute number of report steps.
        # TODO: This might not work all the time, sometimes it might need to be rounded
        # up?
        self.num_timesteps: int = math.floor(
            runspecs["constants"]["INJECTION_TIME"]
            / runspecs["constants"]["REPORTSTEP_LENGTH"]
        )

        # The well cell and the pore volume cell get disregarded in the final dataset.
        # NOTE: Because two cells smaller than well diameter get disregarded when
        # creating the grid, the actual number of xcells is one less than specified in
        # the pyopmnearwell deck.
        # Accounting for this and disregarding the aforementioned cells, substract 4 to
        # get the actual number of cells with valuable data.
        self.num_xcells: int = runspecs["constants"]["NUM_XCELLS"] - 4
        # Fine-scale simulations have a a single layer and are 2D.
        self.num_zcells: int = 1
        self.num_layers: int = 1

        self.data: np.ndarray = data.reshape(
            -1,
            self.num_timesteps,
            self.num_layers,
            self.num_zcells,
            # Add two to account for well and pore volume cell.
            self.num_xcells + 2,
            data_dim,
        )
        # Disregard the pore volume cells. The well cells get disregarded later.
        self.data = self.data[..., :-1, :]

        # Find out the actual number of ensemble members that ran until the end.
        self.num_members: int = data.shape[0]
        self.single_feature_shape: tuple[int, int, int, int] = (
            self.num_members,
            self.num_timesteps,
            self.num_layers,
            self.num_xcells,
        )

        self.runspecs: dict[str, Any] = runspecs

        self.angle: float = angle

    def create_ds(
        self,
        ensemble_dirname: pathlib.Path,
        step_size_x: int = 1,
        step_size_t: int = 1,
        calc_analytical_WI: bool = False,
        log_WI: bool = False,
        log_geom_WI: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create dataset by collecting and upscaling features and target.

        Args:
            ensemble_dirname (pathlib.Path): The directory path where the ensemble
                data is stored. Needed to access ``.GRID`` file.
            step_size_x (int, optional): Step size for reducing the size of data
                along the x-axis. Defaults to 1.
            step_size_t (int, optional): Step size for reducing the size of data
                along the time axis. Defaults to 1.
            calc_analytical_WI (bool, optional): Flag indicating whether to
                calculate analytical WI. Defaults to False.
            log_WI (boo, optional): Flag indicating to take the log (base 10) of the WI
                as target. Defaults to False.
            log_WI (boo, optional): Flag indicating to take the log (base 10) of the
                geometrical part of the WI as feature. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the processed feature data
                and the target data.

        Raises:
            FileNotFoundError: If the preprocessing file is not found.

        """
        feature_lst: list[np.ndarray] = []

        # Get radii from the ensemble simulation. The correction from triangle to radial
        # grid is already applied (in ``ensemble.calculate_radii``).
        cell_center_radii, cell_boundary_radii = self.get_radii(
            ensemble_dirname / "runfiles_0" / "preprocessing" / "GRID.INC"
        )

        # Get all data and cut well cell and pore volume cell (done automatically by the
        # functions).
        # Get pressures.
        feature_lst.append(self.get_vertically_averaged_values(self.data, 0))
        assert feature_lst[-1].shape == self.single_feature_shape, (
            "Pressure feature has wrong shape."
        )

        # Get geometrical part of WI.
        permeabilities: np.ndarray = self.get_homogeneous_values(self.data, 2)
        cell_heights: np.ndarray = self.get_homogeneous_values(self.data, 3)
        feature_lst.append(
            self.get_analytical_PI(
                permeabilities=permeabilities,
                cell_heights=cell_heights,
                radii=cell_center_radii,
                well_radius=cell_boundary_radii[
                    0
                ],  # The innermost well cell was disregarded for the radii. Well radius
                # is the inner radius of the first cell.
            )
        )
        assert feature_lst[-1].shape == self.single_feature_shape, (
            "Geometrical part of WI feature has wrong shape."
        )

        # Get total injected gas. Multiply by 6 to account for cake model of 60°.
        feature_lst.append(
            self.get_homogeneous_values(self.data, 4) * ((math.pi * 2) / self.angle)
        )
        assert feature_lst[-1].shape == self.single_feature_shape, (
            "Total injected volume feature has wrong shape."
        )

        # Get data-driven WI as target. Scaling from cake grid to full 360° grid and
        # from per day to per second takes place inside the function.
        WI_data: np.ndarray = self.get_data_WI(
            self.data,
            0,
            1,
        )
        assert WI_data.shape == self.single_feature_shape, "WI target has wrong shape."

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

        # NOTE For the following, the MLNearWellConfig file is assumed to have been
        # created by pyopmnearwell.ml.nn.scale_and_prepare_dataset in the nn directory.

        # Take logarithms.
        if log_geom_WI:
            feature_lst[1] = np.log10(feature_lst[1])

            # Update the config to reflect the log10 transform for the analytical PI
            # feature.
            config_file = ensemble_dirname.parent / "nn" / "MLNearWellConfig.json"
            if not config_file.exists() or config_file.stat().st_size == 0:
                config = {}
            else:
                with config_file.open("r", encoding="utf-8") as f:
                    config = json.load(f)

            with config_file.open("w", encoding="utf-8") as f:
                config["features"]["inputs"]["ANALYTICAL_PI"]["transform"] = "log10"
                json.dump(config, f, indent=4)

        if log_WI:
            WI_data = np.log10(WI_data)

            # Update the config to reflect the log10 transform for the WI target.
            config_file = ensemble_dirname.parent / "nn" / "MLNearWellConfig.json"
            if not config_file.exists() or config_file.stat().st_size == 0:
                config = {}
            else:
                with config_file.open("r", encoding="utf-8") as f:
                    config = json.load(f)

            with config_file.open("w", encoding="utf-8") as f:
                config["features"]["outputs"]["WI"]["transform"] = "log10"
                json.dump(config, f, indent=4)

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
