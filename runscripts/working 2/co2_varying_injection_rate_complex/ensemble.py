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
    runspecs: dict[str, Any], ensemble_dirname: str | pathlib.Path, **kwargs
) -> np.ndarray:
    """_summary_

    _extended_summary_

    Args:
        runspecs (dict[str, Any]): _description_
        ensemble_dirname (str | pathlib.Path): _description_
        **kwargs: Is passed to ``setup_ensemble`` and ``run_ensemble``. Possible
            parameters are:

    Returns:
        np.ndarray:

    """
    ensemble_dirname = pathlib.Path(ensemble_dirname)

    co2_ensemble = ensemble.create_ensemble(
        runspecs,
        efficient_sampling=[f"inj_rate_{i}" for i in range(5)]
        + [f"INT_inj_time_{i}" for i in range(5)]
        + [f"INT_shut_wells_{i}" for i in range(5)]
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
        # Injection time is 60 days in total (15 - 5 - 15 - 5 - 15 - 5).
        num_report_steps=60 * 10,
        **kwargs,
    )
    features: np.ndarray = np.array(
        ensemble.extract_features(
            data,
            keywords=["PRESSURE", "SGAS", "PERMX", "FLOGASI+", "FGIT"],
            # Outputs are in [mD] and [bar], but OPM uses [m^2] and [Pa] internally.
            keyword_scalings={
                "PRESSURE": units.BAR_TO_PASCAL,
                "PERMX": units.MILIDARCY_TO_M2,
            },
        )
    )
    return features


class Dataset(BaseDataset):
    """Provide methods to create a dataset from ensemble dataset.

    Features are, in the following order:
    1. Pressure - per cell; unit [Pa]
    2. Saturation - per cell; no unit
    3. Permeability - per cell; unit [m^2]
    4. Horizontal flow rate at connection - per layer; unit []
    5. Radii - ; unit [m]
    6. Total injected gas; unit [m^3]
    7. Analytical PI - per cell; unit [...]

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
        # Injection time is 60 days in total (15 - 5 - 15 - 5 - 15 - 5). Only one report
        # step per day was extracted.
        self.num_timesteps: int = 60

        self.num_layers: int = runspecs["constants"]["NUM_LAYERS"]
        self.num_zcells: int = int(
            runspecs["constants"]["NUM_ZCELLS"] / runspecs["constants"]["NUM_LAYERS"]
        )
        # The well cell and the pore volume cell get disregarded.
        self.num_xcells: int = runspecs["constants"]["NUM_XCELLS"] - 2

        self.features: np.ndarray = features.reshape(
            -1,
            self.num_timesteps,
            self.num_layers,
            self.num_zcells,
            self.num_xcells + 2,  # The well cells get disregarded later.
            num_features,
        )
        # Disregard the pore volume cells.
        self.features = self.features[..., :-1, :]

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
        self,
        ensemble_dirname: pathlib.Path,
        step_size_x: int = 1,
        step_size_t: int = 1,
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

        cell_center_radii, cell_boundary_radii = self.get_radii(
            ensemble_dirname / "runfiles_0" / "preprocessing" / "GRID.INC"
        )

        # Cut timesteps where injection is shut. In ``run_ensemble``, the time steps
        # were already reduced s.t. only one data point per day was extracted.
        first_injection_features: np.ndarray = self.features[:, :15, ...]
        second_injection_features: np.ndarray = self.features[:, 20:35, ...]
        third_injection_features: np.ndarray = self.features[:, 40:55, ...]
        self.features = np.concatenate(
            [
                first_injection_features,
                second_injection_features,
                third_injection_features,
            ],
            axis=1,
        )
        # Adjust ``num_timesteps`` accordingly.
        self.num_timesteps = 45

        # Save saturations over the entire domain to integrate later on.
        full_saturations: np.ndarray = self.features[..., 1][..., None]

        # Cut features at radii corresponding to block sizes equal the size of the
        # radial simulation. The averaged values will be wrong for larger values.
        cell_sizes: np.ndarray = formulas.cell_size(
            cell_boundary_radii * formulas.pyopmnearwell_correction()
        )
        self.num_xcells = int(
            np.max(np.nonzero(cell_sizes <= self.runspecs["constants"]["LENGTH"]))
        )
        self.features = self.features[..., : self.num_xcells + 1, :]

        # Cut ``cell_center_radii`` but not ``cell_boundary_radii``. The latter is still
        # needed in full for integration of saturation.
        cell_center_radii = cell_center_radii[: self.num_xcells]

        # Update single_feature_shape`` s.t. all assertions still work.
        self.single_feature_shape = (
            self.num_members,
            self.num_timesteps,
            self.num_layers,
            self.num_xcells,
        )

        # Get all data.
        # Get pressures
        feature_lst.append(self.get_vertically_averaged_values(self.features, 0))

        # Get saturations
        feature_lst.append(
            self.get_horizontically_integrated_values(
                full_saturations,
                # Integrate only for center radii small enough. The full
                # full_saturations and cell_boundary_radii arrays need to be passed,
                # s.t. saturation can be integrated over the grid block corresponding to
                # the equivalent well radius.
                cell_center_radii,
                cell_boundary_radii,
                0,
            )
        )

        # Get permeabilities
        feature_lst.append(self.get_homogeneous_values(self.features, 2))

        # Get injection rates
        flow_rates_i: np.ndarray = self.get_vertically_averaged_values(
            self.features, 3, disregard_first_xcell=False
        )
        # Extend by one axis to enable broadcasting later on.
        injection_rates: np.ndarray = flow_rates_i[..., 0][..., None]
        feature_lst.append(injection_rates)

        # Get equivalent well radii
        feature_lst.append(cell_center_radii)
        # Get total injected volume. Multiply by 6 to account for cake model.
        feature_lst.append(self.get_homogeneous_values(self.features, 4) * 6)
        # Get analytical PI
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

        return (
            np.stack(feature_lst, axis=-1),
            WI_data,
        )


def main():
    ensemble_dirname: pathlib.Path = dirname / runspecs_ensemble["name"]
    data_dirname: pathlib.Path = dirname / f"dataset_{runspecs_ensemble['name']}"

    ensemble_dirname.mkdir(exist_ok=True)
    data_dirname.mkdir(exist_ok=True)

    extracted_data: np.ndarray = full_ensemble(
        runspecs_ensemble, ensemble_dirname, step_size_time=10
    )
    np.save(str(ensemble_dirname / "features"), extracted_data)

    extracted_data = np.load(str(ensemble_dirname / "features.npy"))
    dataset = Dataset(extracted_data, runspecs_ensemble, 5)
    # We set the step size for x cells here. In full_ensemble it requires some elaborate
    # thinking since the cells are flattened.
    features, targets = dataset.create_ds(
        ensemble_dirname, step_size_x=2, step_size_t=1
    )
    ensemble.store_dataset(features, targets, data_dirname)


if __name__ == "__main__":
    main()
