from __future__ import annotations

import math
import pathlib
from typing import Any

import numpy as np
from pyopmnearwell.ml.upscale import BaseUpscaler
from pyopmnearwell.utils import formulas
from pyopmnearwell.utils import units

dirname: pathlib.Path = pathlib.Path(__file__).parent


class CO2_3D_upscaler(BaseUpscaler):
    """Upscale ensemble data from 3D CO2 simulations to coarse cartesian cells and
    arange into ``np.ndarrays`` to be used in a ``tensorflow`` dataset.

    Features are, in the following order:
    1. Pressure - per cell; unit [Pa]
    2. Saturation - per cell; no unit
    3. Permeability - per cell; unit [m^2]
    4. Radii - ; unit [m]
    5. Total injected gas; unit [m^3]
    6. Geometrical part of WI - per cell; unit [...]
    (7. Analytical WI - per cell; unit [...] -> This is just for comparison and not used
        during training.)

    Input data from the ensemble is assumed to be, in the following order:
    1. PRESSURE - per cell; unit [Pa]
    2. SGAS - per cell; unit [%]
    3. FLOGASI+ - per cell; unit [m^3/s] ??
    4. PERMX - per cell; unit [m^2]
    5. DZ - per cell; unit [m]
    6. FGIT - global; unit [m^3]

    The feature array will have shape ``(num_ensemble_runs, num_timesteps/step_size_t,
    num_layers, num_xcells/step_size_x, num_features)``.
    The target array will have shape ``(num_ensemble_runs, num_timesteps/step_size_t,
    num_layers, num_xcells/step_size_x, 1)``

    """

    def __init__(
        self,
        data: np.ndarray,
        runspecs: dict[str, Any],
        data_dim: int = 5,
        angle: float = math.pi / 3,
    ) -> None:
        """_summary_

        Note: It is always assumed that if reshaped with ... ordering, the features
        array will have the following structure:

        ``shape == (num_ensemble_runs, num_timesteps, num_layers, num_zcells_per_layer,
        num_xcells, features)``

        Args:
            data (np.ndarray): Data from the ensemble simulation. See above.
            runspecs (dict[str, Any]): The runspecs dictionary from the ensemble.
            data_dim (int): Dimension of a single data point. Should not be changed
                unless for very good reason. Default is 6.
            angle (float): Angle between both sides of the triangle grid.
                Default is ``math.pi / 3``.

        """
        # TODO: This might not work all the time, sometimes it might need to be rounded
        # up?
        self.num_timesteps: int = math.floor(
            runspecs["constants"]["INJECTION_TIME"]
            / runspecs["constants"]["REPORTSTEP_LENGTH"]
        )

        self.num_layers: int = runspecs["constants"]["NUM_LAYERS"]
        self.num_zcells: int = int(
            runspecs["constants"]["NUM_ZCELLS"] / runspecs["constants"]["NUM_LAYERS"]
        )
        # The well cell and the pore volume cell get disregarded.
        # NOTE: Because two cells smaller than well diameter get disregarded when
        # creating the grid, the actual number of xcells is one less than specified in
        # the pyopmnearwell deck.
        # Accounting for this and disregarding the aforementioned cells, substract 4 to
        # get the actual number of cells with valuable data.
        self.num_xcells: int = runspecs["constants"]["NUM_XCELLS"] - 4

        self.data: np.ndarray = data.reshape(
            -1,
            self.num_timesteps,
            self.num_layers,
            self.num_zcells,
            self.num_xcells + 2,  # The well cells get disregarded later.
            data_dim,
        )
        # Disregard the pore volume cells.
        self.data = self.data[..., :-1, :]

        # Find out the actual number of ensemble members that ran until the end.
        self.num_members: int = self.data.shape[0]
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

        self.angle: float = angle

    def get_data_WI_safe(
        self,
        data: np.ndarray,
        pressure_index: int,
        flow_index: int,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Robust WI computation that does not crash for shut-in (dp ~ 0),
        and returns shape == self.single_feature_shape.

        Assumes:
        - x=0 is the well cell (proxy for BHP)
        - x=1: are the reservoir cells we keep
        - vertical axis (zcells) must be reduced away
        """
        # Full fields: (..., zcells, xcells_plus_well)
        p_full = data[..., pressure_index]
        q_full = data[..., flow_index]

        # Well cell pressure used as "bhp" proxy (still has zcells)
        bhp = p_full[..., 0]                 # (..., zcells)

        # Reservoir cells (drop well cell) -> (..., zcells, xcells)
        p = p_full[..., 1:]
        q = q_full[..., 1:]

        # dp per zcell and xcell
        dp = bhp[..., None] - p              # (..., zcells, xcells)

        # Reduce vertical axis (zcells). Use the same strategy as pressures:
        # average over zcells to get one WI per layer and xcell.
        dp = np.average(dp, axis=-2)         # (..., xcells)
        q = np.average(q, axis=-2)           # (..., xcells)

        # Clamp dp away from zero (shut-in can give dp == 0)
        dp = np.where(np.abs(dp) < eps, eps, dp)

        WI = q / dp                          # (..., xcells)
        return WI

    def create_ds(
        self,
        ensemble_dirname: pathlib.Path,
        step_size_x: int = 1,
        step_size_t: int = 1,
        calc_analytical_WI: bool = False,
        keep_xcells: int | None = None,   # <-- NY
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create dataset by collecting and upscaling features and target.

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

        # Get radii from the ensemble simulation. The correction from triangle to radial
        # grid is already applied (in ``ensemble.calculate_radii``).
        cell_center_radii, cell_boundary_radii = self.get_radii(
            ensemble_dirname / "runfiles_0" / "preprocessing" / "GRID.INC"
        )

        # Save saturations over the entire domain to integrate later on.
        full_saturations: np.ndarray = self.data[..., 1][..., None]

        # Cut features at equivalent radii corresponding to block sizes equal the size
        # of the radial simulation. The averaged values would be wrong for larger
        # values.
        # NOTE: "LENGTH" in the ensemble simulation correpsonds to twice the block size
        # for an upscaled cell, because it is symmetric around the well.
        # NOTE: To upscale the saturation for, e.g., a cell of size 100x100m, one needs
        # radial values for up to (50/cos(45°))m = sqrt(2)*50m. Thus the maximum
        # possible equivalent cell size is "LENGTH" * sqrt(2).

        #ENDRET 
        cell_sizes = formulas.cell_size(cell_boundary_radii)

        # eksisterende cut basert på LENGTH
        length_cut = int(
            np.max(
                np.nonzero(
                    cell_sizes <= self.runspecs["constants"]["LENGTH"] * math.sqrt(2)
                )
            )
        )

        # NY: overstyr til bare nærbrønn for treningsdata
        if keep_xcells is not None:
            self.num_xcells = min(length_cut, keep_xcells)
        else:
            self.num_xcells = length_cut

        # kutt data i x-retning (merk +1 pga well-cell håndtering i dataen)
        self.data = self.data[..., : self.num_xcells + 1, :]

        # kutt radii tilsvarende
        cell_center_radii = cell_center_radii[: self.num_xcells]
        #ENDRET SLUTT
        # Update single_feature_shape`` s.t. all assertions still work.
        self.single_feature_shape = (
            self.num_members,
            self.num_timesteps,
            self.num_layers,
            self.num_xcells,
        )

        # Get all data.
        # Get pressures. Average vertically over all cells inside a layer.
        feature_lst.append(self.get_vertically_averaged_values(self.data, 0))
        assert (
            feature_lst[-1].shape == self.single_feature_shape
        ), "Pressures feature has wrong shape."
        # Get saturations. Sum over all cells inside a layer and integrate over all
        # horizontical cells.
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
        assert (
            feature_lst[-1].shape == self.single_feature_shape
        ), "Saturations feature has wrong shape."
        # Get equivalent well radii (broadcasted later)
        feature_lst.append(cell_center_radii)

        # Total injected volume (FGIT), cake model factor
        feature_lst.append(self.get_homogeneous_values(self.data, 3) * 6)
        assert feature_lst[-1].shape == self.single_feature_shape, \
            "Total injected volume feature has wrong shape."

        # Injection rate (WGIR:INJ0) -> ligger på indeks 4
        feature_lst.append(self.get_homogeneous_values(self.data, 4) * 6)
        assert feature_lst[-1].shape == self.single_feature_shape, \
            "Total injected volume feature has wrong shape."

        # Constant cell height [m]
        cell_heights = np.full(self.single_feature_shape, 5.0, dtype=float)

        # Constant permeability [mD]
        perm_const = 2e-13 * units.M2_TO_MILIDARCY
        permeabilities = np.full(self.single_feature_shape, perm_const, dtype=float)

        # Analytical well index
        feature_lst.append(
            self.get_analytical_PI(
                permeabilities=permeabilities,
                cell_heights=cell_heights,
                radii=cell_center_radii,
                well_radius=cell_boundary_radii[0],
            )
        )
        assert feature_lst[-1].shape == self.single_feature_shape, \
            "Geometrical part of WI feature has wrong shape."

        # Data-driven WI (target)
        WI_data = self.get_data_WI_safe(self.data, 0, 2)
        assert WI_data.shape == self.single_feature_shape, \
            "WI target has wrong shape."

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
                    feature_lst[1],
                    feature_lst[2],
                    self.runspecs["constants"]["INIT_TEMPERATURE"],
                    self.runspecs["constants"]["SURFACE_DENSITY"],
                    radii=feature_lst[3],
                    well_radius=cell_boundary_radii[0],
                    OPM=self.runspecs["constants"]["OPM"],
                )
            )
        #ENDRET!
        # Bygg endelige arrays
        features = np.stack(feature_lst, axis=-1)   # shape: (nmembers, nt, nlayers, nx, nfeat)
        targets  = WI_data                          # shape: (nmembers, nt, nlayers, nx)

        # --- FORCE FIXED NX FOR TRAINING SHAPE ---
        DESIRED_NX = 11

        def force_nx(a: np.ndarray) -> np.ndarray:
            """
            Tving nx=DESIRED_NX langs x-aksen.
            For targets: a.ndim==4, x-aksen er axis=3
            For features: a.ndim==5, x-aksen er axis=3
            """
            x_axis = 3
            nx = a.shape[x_axis]

            if nx > DESIRED_NX:
                sl = [slice(None)] * a.ndim
                sl[x_axis] = slice(0, DESIRED_NX)
                return a[tuple(sl)]

            if nx < DESIRED_NX:
                # pad ved å kopiere siste x-søyle
                sl_last = [slice(None)] * a.ndim
                sl_last[x_axis] = slice(nx - 1, nx)        # siste x-kolonne med bevart dimensjon
                last = a[tuple(sl_last)]

                reps = [1] * a.ndim
                reps[x_axis] = DESIRED_NX - nx
                pad = np.repeat(last, reps[x_axis], axis=x_axis)

                return np.concatenate([a, pad], axis=x_axis)

            return a

        features = force_nx(features)
        targets  = force_nx(targets)
        # --- END FORCE NX ---

        return features, targets

