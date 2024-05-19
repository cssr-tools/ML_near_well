import pathlib
import sys
from typing import Any

import numpy as np
import seaborn as sns
from pyopmnearwell.ml import data, ensemble, integration, nn
from pyopmnearwell.utils import formulas, units
from runspecs import runspecs_ensemble, runspecs_integration, trainspecs
from tensorflow import keras

dirname: pathlib.Path = pathlib.Path(__file__).parent

sys.path.append(str(dirname / ".."))
from utils import full_ensemble, plot_member, tune_and_train

# Set seaborn style.
sns.set_theme()

# Structure directories.
ensemble_dir: pathlib.Path = dirname / "ensemble"
data_dir: pathlib.Path = dirname / "dataset"
nn_dir: pathlib.Path = dirname / "nn"

ensemble_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
nn_dir.mkdir(parents=True, exist_ok=True)

# Get OPM installations.
# TODO: These need to be adjusted for reproducing results.
OPM: pathlib.Path = pathlib.Path("/home/peter/Documents/2023_CEMRACS/opm")
FLOW: pathlib.Path = OPM / "build" / "opm-simulators" / "bin" / "flow"
OPM_ML: pathlib.Path = pathlib.Path("/home/peter/Documents/2023_CEMRACS/opm_ml")
FLOW_ML: pathlib.Path = (
    OPM_ML / "build" / "opm-simulators" / "bin" / "flow_gaswater_dissolution_diffuse"
)


class CO2_2D_Dataset(data.BaseDataset):
    """Provide methods to create a dataset from ensemble dataset.

    Features are, in the following order:
    1. Pressure - per cell; unit [Pa]
    2. Geometrical part of WI; unit [~]
    3. Total injected gas - global; unit [m^3]
    (4. Analytical WI - per cell; unit [...] -> This is just for comparison and not used
        during training.)

    The feature array will have shape ``(num_ensemble_runs, num_timesteps/step_size_t, num_xcells/step_size_x, num_features)``.
    The target array will have shape ``(num_ensemble_runs, num_timesteps/step_size_t, num_xcells/step_size_x, 1)``

    """

    def __init__(
        self, features: np.ndarray, runspecs: dict[str, Any], num_features: int
    ):
        """_summary_

        Note: It is always assumed that if reshaped with ... ordering, the features
        array will have the following structure:

        ``shape == (num_ensemble_runs, num_timesteps, num_layers, num_zcells_per_layer,
        num_xcells, features)``

        """
        # Find out the actual number of ensemble members that ran until the end.
        self.num_members: int = features.shape[0]

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

        self.features: np.ndarray = features.reshape(
            -1,
            self.num_timesteps,
            self.num_layers,
            self.num_zcells,
            self.num_xcells + 2,
            num_features,
        )
        # Disregard the pore volume cells. The well cells get disregarded later.
        self.features = self.features[..., :-1, :]

        # Find out the actual number of ensemble members that ran until the end.
        self.num_members: int = self.features.shape[0]
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

        # Get radii of cell centers from preprocessing file.
        cell_center_radii, cell_boundary_radii = self.get_radii(
            ensemble_dirname / "runfiles_0" / "preprocessing" / "GRID.INC"
        )

        # Cut features at radii corresponding to block sizes equal the size of the
        # radial simulation. The averaged values will be wrong for larger values.
        cell_sizes: np.ndarray = formulas.cell_size(
            cell_boundary_radii * formulas.pyopmnearwell_correction()
        )
        self.num_xcells = int(
            np.max(np.nonzero(cell_sizes <= self.runspecs["constants"]["LENGTH"]))
        )

        # Update ``single_feature_shape`` s.t. all assertions still work.
        self.single_feature_shape = (
            self.num_members,
            self.num_timesteps,
            self.num_layers,
            self.num_xcells,
        )

        # Cut ``self.features`` and ``cell_center_radii`` accordingly.
        self.features = self.features[..., : self.num_xcells + 1, :]
        cell_center_radii = cell_center_radii[: self.num_xcells]

        # Get all data.
        # Get pressures.
        feature_lst.append(self.get_vertically_averaged_values(self.features, 0))
        # Get geometrical part of WI.
        permeabilities: np.ndarray = self.get_homogeneous_values(self.features, 2)
        cell_heights: np.ndarray = self.get_homogeneous_values(self.features, 3)
        feature_lst.append(
            self.get_analytical_PI(
                permeabilities=permeabilities,
                cell_heights=cell_heights,
                radii=cell_center_radii,
                well_radius=cell_boundary_radii[1],
            )
        )
        # Get total injected gas.
        feature_lst.append(self.get_homogeneous_values(self.features, 4))

        # Get data based WI as target.
        WI_data: np.ndarray = self.get_data_WI(
            self.features,
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

        return (
            np.stack(feature_lst, axis=-1),
            WI_data,
        )


def main() -> None:
    # Run ensemble and create data.
    if True:
        extracted_data: np.ndarray = full_ensemble(
            runspecs_ensemble,
            ensemble_dir,
            ecl_keywords=["PRESSURE", "FLOGASI+"],
            init_keywords=["PERMX", "DZ"],
            summary_keywords=["FGIT"],
            recalc_grid=True,
            recalc_sections=True,
            recalc_tables=False,
        )
        np.save(str(ensemble_dir / "features"), extracted_data)

    # Setup dataset for training.
    extracted_data = np.load(str(ensemble_dir / "features.npy"))
    dataset = CO2_2D_Dataset(extracted_data, runspecs_ensemble, 5)
    features, targets = dataset.create_ds(ensemble_dir, step_size_x=1, step_size_t=3)

    # Remove WI_analytical for training.
    if False:
        train_features: np.ndarray = features[..., :-1]
    # Flatten features and targets before storing.
    ensemble.store_dataset(
        np.reshape(features, (-1, 4)),
        np.reshape(targets, (-1, 1)),
        data_dir,
    )

    # Plot some WI.s
    for i in range(0, runspecs_ensemble["npoints"], 30):
        plot_member(
            features,
            targets,
            i,
            data_dir / f"member_{i}_WI_vs_radius",
            comparison_param="layer",
            fixed_param_index=10,  # Plot for time step 10.
            radius_index=3,
            permeability_index=1,
        )
        plot_member(
            features,
            targets,
            i,
            data_dir / f"member_{i}_WI_vs_time",
            x_param="time",
            comparison_param="layer",
            final_time=runspecs_ensemble["constants"]["INJECTION_TIME"],
            fixed_param_index=10,  # Plot for time step 10.
            radius_index=3,
            permeability_index=1,
        )

    # Train model
    if True:
        tune_and_train(trainspecs, data_dir, nn_dir)
    model: keras.Model = keras.models.load_model(nn_dir / "bestmodel.keras")
    for i in range(0, runspecs_ensemble["npoints"], 30):
        plot_member(
            features,
            targets,
            i,
            nn_dir / f"member_{i}_WI_vs_radius",
            comparison_param="layer",
            fixed_param_index=10,  # Plot for time step 10.
            radius_index=3,
            permeability_index=1,
            model=model,
            nn_dirname=nn_dir,
            trainspecs=trainspecs,
        )
        plot_member(
            features,
            targets,
            i,
            nn_dir / f"member_{i}_WI_vs_time",
            x_param="time",
            comparison_param="layer",
            final_time=runspecs_ensemble["constants"]["INJECTION_TIME"],
            fixed_param_index=10,  # Plot for time step 10.
            radius_index=3,
            permeability_index=1,
            model=model,
            nn_dirname=nn_dir,
            trainspecs=trainspecs,
        )

    # Integrate
    if True:
        pass


if __name__ == "__main__":
    main()
