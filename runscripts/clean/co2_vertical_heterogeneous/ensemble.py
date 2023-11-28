import os
import pathlib
from typing import Any

import numpy as np
from pyopmnearwell.ml import ensemble
from pyopmnearwell.ml.data import BaseDataset
from pyopmnearwell.utils import formulas, units
from runspecs import runspecs_ensemble

dirname: pathlib.Path = pathlib.Path(__file__).parent


def full_ensemble(
    runspecs: dict[str, Any], ensemble_dirname: str | pathlib.Path
) -> np.ndarray:
    """_summary_

    _extended_summary_

    Args:
        runspecs (dict[str, Any]): _description_
        dirname (str | pathlib.Path): _description_

    Returns:
        np.ndarray:

    """
    ensemble_dirname = pathlib.Path(ensemble_dirname)

    co2_ensemble = ensemble.create_ensemble(
        runspecs,
        efficient_sampling=[
            f"PERM_{i}" for i in range(runspecs["constants"]["NUM_ZCELLS"])
        ]
        + ["INIT_PRESSURE"],
    )
    ensemble.setup_ensemble(
        ensemble_dirname,
        co2_ensemble,
        ensemble_dirname / ".." / "ensemble.mako",
        recalc_grid=False,
        recalc_sections=True,
        recalc_tables=False,
    )
    data: dict[str, Any] = ensemble.run_ensemble(
        runspecs["constants"]["FLOW"],
        ensemble_dirname,
        runspecs,
        ecl_keywords=["PRESSURE", "SGAS", "FLOGASI+"],
        init_keywords=["PERMX"],
        summary_keywords=["FGIT"],
        num_report_steps=runspecs["constants"]["INJECTION_TIME"] * 10,
    )
    features: np.ndarray = np.array(
        ensemble.extract_features(
            data,
            keywords=["PRESSURE", "SGAS", "PERMX", "FLOGASI+", "FGIT"],
            keyword_scalings={"PRESSURE": units.BAR_TO_PASCAL},
        )
    )
    return features


class Dataset(BaseDataset):
    """Provide methods to create a dataset from ensemble dataset.

    Features are, in the following order:
    1. Pressure - per cell; unit [Pa]
    2. Saturation - per cell; no unit
    3. Permeability - per cell; unit [mD]
    4. Radii - ; unit [m]
    5. Total injected gas; unit [m^3]
    6. Analytical PI - per cell; unit [...]
    7. Analytical PI - per cell; unit [...] -> This is just for comparison and not used
        during training.

    The feature array will have shape ``(num_ensemble_runs, num_timesteps/step_size_t,
    num_layers, num_xcells/step_size_x, num_features)``.
    The target array will have shape ``(num_ensemble_runs, num_timesteps/step_size_t,
    num_layers, num_xcells/step_size_x, 1)``

    """

    def __init__(
        self,
        features: np.ndarray,
        runspecs: dict[str, Any],
        num_features: int,
    ) -> None:
        """_summary_

        Note: It is always assumed that if reshaped with ... ordering, the features
        array will have the following structure:

        ``shape == (num_ensemble_runs, num_timesteps, num_layers, num_zcells_per_layer,
        num_xcells, features)``

        """
        self.num_timesteps: int = runspecs["constants"]["INJECTION_TIME"] * 10

        self.num_layers: int = runspecs["constants"]["NUM_LAYERS"]
        self.num_zcells: int = int(
            runspecs["constants"]["NUM_ZCELLS"] / runspecs["constants"]["NUM_LAYERS"]
        )
        self.num_xcells: int = runspecs["constants"]["NUM_XCELLS"] - 1

        self.features: np.ndarray = features.reshape(
            -1,
            self.num_timesteps,
            self.num_layers,
            self.num_zcells,
            self.num_xcells + 1,  # The well cells get disregarded later.
            num_features,
        )
        # Find out the actual number of ensemble members that ran until the end.
        self.num_members: int = self.features.shape[0]
        self.single_feature_shape: tuple[int, int, int, int] = (
            self.num_members,
            self.num_timesteps,
            self.num_layers,
            self.num_xcells,
        )
        # NOTE: Each data array is reshaped into ``(num_members, num_report_steps, num_layers,
        # num_zcells/num_layers, num_xcells).
        #   - The second to last axis (i.e., all vertical cells in one layer) is eliminated by
        #     averaging or summation.
        #   - The last axis (i.e., all horizontal cells in one layer) is eliminated by
        #     integration or picking a single value.
        self.runspecs: dict[str, Any] = runspecs

    def create_ds(
        self, ensemble_dirname: pathlib.Path, step_size_x: int = 1, step_size_t: int = 1
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
        # Get pressures
        feature_lst.append(self.get_vertically_averaged_values(self.features, 0))

        cell_center_radii, cell_boundary_radii = self.get_radii(
            ensemble_dirname / "runfiles_0" / "preprocessing" / "GRID.INC"
        )
        # Get saturations
        feature_lst.append(
            self.get_horizontically_integrated_values(
                self.features, cell_center_radii, cell_boundary_radii, 1
            )
        )
        # Get permeabilities
        feature_lst.append(self.get_homogeneous_values(self.features, 2))
        feature_lst.append(cell_center_radii)
        feature_lst.append(self.get_homogeneous_values(self.features, 4))
        feature_lst.append(
            self.get_analytical_PI(
                feature_lst[2],
                radii=cell_center_radii,
                well_radius=cell_boundary_radii[0],
            )
        )

        # assert cell_boundary_radii[0] == 0.25
        WI_data: np.ndarray = self.get_data_WI(
            self.features,
            0,
            3,
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
        # impossible.
        feature_lst.append(
            self.get_analytical_WI(
                feature_lst[0],
                feature_lst[1],
                feature_lst[2],
                self.runspecs["constants"]["INIT_TEMPERATURE"],
                self.runspecs["constants"]["SURFACE_DENSITY"],
                radii=feature_lst[3],
                well_radius=cell_boundary_radii[0],
                OPM=self.runspecs["constants"]["OPM"],
            )
        )

        return (
            np.stack(feature_lst, axis=-1),
            WI_data,
        )


if __name__ == "__main__":
    ensemble_dirname: pathlib.Path = dirname / runspecs_ensemble["name"]
    data_dirname: pathlib.Path = dirname / f"dataset_{runspecs_ensemble['name']}"

    ensemble_dirname.mkdir(exist_ok=True)
    data_dirname.mkdir(exist_ok=True)

    extracted_data: np.ndarray = full_ensemble(runspecs_ensemble, ensemble_dirname)
    np.save(str(ensemble_dirname / "features"), extracted_data)

    dataset = Dataset(extracted_data, runspecs_ensemble, 5)
    features, targets = dataset.create_ds(
        ensemble_dirname, step_size_x=5, step_size_t=5
    )
    ensemble.store_dataset(features, targets, data_dirname)
