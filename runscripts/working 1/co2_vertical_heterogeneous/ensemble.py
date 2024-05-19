import math
import os
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from pyopmnearwell.ml import ensemble
from pyopmnearwell.utils import formulas, plotting, units
from runspecs import runspecs_ensemble_1 as runspecs_ensemble

dirname: str = os.path.dirname(__file__)

ensemble_dirname: str = os.path.join(dirname, runspecs_ensemble["name"])
data_dirname: str = os.path.join(dirname, f"dataset_{runspecs_ensemble['name']}")

os.makedirs(ensemble_dirname, exist_ok=True)
os.makedirs(data_dirname, exist_ok=True)


################
# Run ensemble #
################
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

###############
# Get dataset #
###############
features: np.ndarray = np.array(
    ensemble.extract_features(
        data,
        keywords=["PRESSURE", "SGAS", "PERMX", "FLOGASI+", "FGIT"],
        keyword_scalings={"PRESSURE": units.BAR_TO_PASCAL},
    )
)
num_completed_runs: int = features.shape[0]

# NOTE: Each data array is reshaped into ``(num_members, num_report_steps, num_layers,
# num_zcells/num_layers, num_xcells).
#   - The second to last axis (i.e., all vertical cells in one layer) is eliminated by
#     averaging or summation.
#   - The last axis (i.e., all horizontal cells in one layer) is eliminated by
#     integration or picking a single value.

#############
# Pressures #
#############
# The pressure value in the well block is approximately equal to the pressure at
# equivalent well radius. For a grid block size of 100x100m the equivalent well radius
# is r_e=20.07m. On a triangle grid with theta = pi/6 this is approximately equivalent
# to to an altidude of x = 20.07/1.1 = 18.2015m which is the 15th fine scale cell.
# TODO: Make the 15th cell thingy more flexible.
pressures: np.ndarray = np.average(
    features[..., 0].reshape(
        features.shape[0],
        features.shape[1],
        runspecs_ensemble["constants"]["NUM_LAYERS"],
        int(
            runspecs_ensemble["constants"]["NUM_ZCELLS"]
            / runspecs_ensemble["constants"]["NUM_LAYERS"]
        ),
        -1,
    ),
    axis=-2,
)[
    ..., 15
]  # ``shape = (num_completed_runs, num_timesteps/3, num_layers); unit [Pa]
assert pressures.shape == (
    num_completed_runs,
    runspecs_ensemble["constants"]["INJECTION_TIME"] * 10,
    runspecs_ensemble["constants"]["NUM_LAYERS"],
)


###############
# Saturations #
###############
# Get full list of cell boundary radii.
_, inner_radii, outer_radii = ensemble.calculate_radii(
    os.path.join(ensemble_dirname, "runfiles_0", "preprocessing", "GRID.INC"),
    num_cells=40,
    return_outer_inner=True,
    triangle_grid=True,
    theta=math.pi / 3,
)
radii: np.ndarray = np.append(inner_radii, outer_radii[-1])

saturations: np.ndarray = np.sum(
    features[..., 1].reshape(
        features.shape[0],
        features.shape[1],
        runspecs_ensemble["constants"]["NUM_LAYERS"],
        int(
            runspecs_ensemble["constants"]["NUM_ZCELLS"]
            / runspecs_ensemble["constants"]["NUM_LAYERS"]
        ),
        -1,
    ),
    axis=-2,
)
# Integrate saturation along layers and divide by block volume.
saturations = ensemble.integrate_fine_scale_value(
    saturations, radii, block_sidelength=runspecs_ensemble["constants"]["LENGTH"]
) / (
    runspecs_ensemble["constants"]["LENGTH"] ** 2
    * (
        runspecs_ensemble["constants"]["HEIGHT"]
        / runspecs_ensemble["constants"]["NUM_LAYERS"]
    )
)
assert saturations.shape == (
    num_completed_runs,
    runspecs_ensemble["constants"]["INJECTION_TIME"] * 10,
    runspecs_ensemble["constants"]["NUM_LAYERS"],
)

##################
# Permeabilities #
##################
# The permeabilities are equal for an entire layer. -> Take the first cell value
permeabilities: np.ndarray = features[..., 2].reshape(
    features.shape[0],
    features.shape[1],
    runspecs_ensemble["constants"]["NUM_LAYERS"],
    -1,
)[
    ..., 0
]  # ``shape = (num_completed_runs, num_timesteps/3, num_layers); unit [mD]
assert permeabilities.shape == (
    num_completed_runs,
    runspecs_ensemble["constants"]["INJECTION_TIME"] * 10,
    runspecs_ensemble["constants"]["NUM_LAYERS"],
)

#############
# Timesteps #
#############
timesteps: np.ndarray = np.linspace(
    0, runspecs_ensemble["constants"]["INJECTION_TIME"], features.shape[-3]
)  # unit: [day]
assert timesteps.shape == (runspecs_ensemble["constants"]["INJECTION_TIME"] * 10,)

######################
# Total injected gas #
######################
# Is equal for all layers. -> Take the first value for each layer.
tot_inj_gas: np.ndarray = features[..., 4].reshape(
    features.shape[0],
    features.shape[1],
    runspecs_ensemble["constants"]["NUM_LAYERS"],
    -1,
)[
    ..., 0
]  # unit: [...]
assert tot_inj_gas.shape == (
    num_completed_runs,
    runspecs_ensemble["constants"]["INJECTION_TIME"] * 10,
    runspecs_ensemble["constants"]["NUM_LAYERS"],
)


###########################
# Multi-phase Peaceman WI #
###########################
densities_lst: list[list[float]] = []
viscosities_lst: list[list[float]] = []
for pressure in pressures.flatten():
    # Evaluate density and viscosity.
    density_tuple: list[float] = []
    viscosity_tuple: list[float] = []

    for phase in ["water", "CO2"]:
        density_tuple.append(
            formulas.co2brinepvt(
                pressure=pressure,
                temperature=runspecs_ensemble["constants"]["INIT_TEMPERATURE"]
                + units.CELSIUS_TO_KELVIN,
                property="density",
                phase=phase,
            )
        )

        viscosity_tuple.append(
            formulas.co2brinepvt(
                pressure=pressure,
                temperature=runspecs_ensemble["constants"]["INIT_TEMPERATURE"]
                + units.CELSIUS_TO_KELVIN,
                property="viscosity",
                phase=phase,
            )
        )
    densities_lst.append(density_tuple)
    viscosities_lst.append(viscosity_tuple)

densities_shape = list(pressures.shape)
densities_shape.extend([2])
densities: np.ndarray = np.array(densities_lst).reshape(densities_shape)
viscosities: np.ndarray = np.array(viscosities_lst).reshape(densities_shape)

# Calculate the well index from Peaceman. The analytical well index is in [m*s],
# hence we need to devide by surface density to transform to [m^4*s/kg].
WI_analytical: np.ndarray = (
    formulas.two_phase_peaceman_WI(
        k_h=permeabilities
        * units.MILIDARCY_TO_M2
        * runspecs_ensemble["constants"]["HEIGHT"]
        / runspecs_ensemble["constants"]["NUM_LAYERS"],
        r_e=formulas.equivalent_well_block_radius(
            runspecs_ensemble["constants"]["LENGTH"]
        ),
        r_w=0.25,
        rho_1=densities[..., 0],
        mu_1=viscosities[..., 0],
        k_r1=(1 - saturations) ** 2,
        rho_2=densities[..., 1],
        mu_2=viscosities[..., 1],
        k_r2=saturations**2,
    )
    / runspecs_ensemble["constants"]["SURFACE_DENSITY"]
)

assert WI_analytical.shape == (
    num_completed_runs,
    runspecs_ensemble["constants"]["INJECTION_TIME"] * 10,
    runspecs_ensemble["constants"]["NUM_LAYERS"],
)

############################
# Data-driven WI as target #
############################
# Take the pressure values of the well blocks as bhp.
bhps: np.ndarray = np.average(
    features[..., 0].reshape(
        features.shape[0],
        features.shape[1],
        runspecs_ensemble["constants"]["NUM_LAYERS"],
        int(
            runspecs_ensemble["constants"]["NUM_ZCELLS"]
            / runspecs_ensemble["constants"]["NUM_LAYERS"]
        ),
        -1,
    ),
    axis=-2,
)[
    ..., 0
]  # ``shape = (num_completed_runs, num_timesteps/3, num_layers); unit [bar]

# Get the individual injection rates per second for each cell. Multiply by 6 to account
# for the 60° cake and transform to rate per second.
injection_rate_per_second_per_cell: np.ndarray = (
    np.average(
        features[..., 3].reshape(
            features.shape[0],
            features.shape[1],
            runspecs_ensemble["constants"]["NUM_LAYERS"],
            int(
                runspecs_ensemble["constants"]["NUM_ZCELLS"]
                / runspecs_ensemble["constants"]["NUM_LAYERS"]
            ),
            -1,
        ),
        axis=-2,
    )[..., 0]
    * 6
    * units.Q_per_day_to_Q_per_seconds
)  # ``shape = (num_completed_runs, num_timesteps/3, num_layers)
# Check that we do not divide by zero.
assert np.all(bhps - pressures)
WI_data: np.ndarray = injection_rate_per_second_per_cell / (bhps - pressures)
assert WI_data.shape == (
    num_completed_runs,
    runspecs_ensemble["constants"]["INJECTION_TIME"] * 10,
    runspecs_ensemble["constants"]["NUM_LAYERS"],
)


# Features are, in the following order:
# 1. Pressure - per cell; unit [Pa]
# 2. Saturation - per cell; no unit
# 3. Permeability - per cell; unit [mD]
# 4. Total injected gas; unit [...]
# 5. Analytical WI - per cell; unit [...]

# shape ``shape = (num_completed_runs, num_timesteps/3, num_layers, 3)``
ensemble.store_dataset(
    np.stack(
        [
            pressures,
            saturations,
            permeabilities,
            np.broadcast_to(tot_inj_gas, pressures.shape),
            WI_analytical,
        ],
        axis=-1,
    ),
    WI_data[..., None],
    data_dirname,
)


############
# Plotting #
############
# Comparison vs. Peaceman for the first, third and last layer. Only first ensemble
# member.
fig_1 = plt.figure(1)
fig_2 = plt.figure(2)
for i, color in zip([0, 2, 4], plt.cm.rainbow(np.linspace(0, 1, 3))):
    pressures_member = pressures[0, ..., i]
    bhp_member = bhps[0, ..., i]
    injection_rate_per_second_per_cell_member = injection_rate_per_second_per_cell[
        0, ..., i
    ]
    WI_data_member = WI_data[0, ..., i]
    WI_analytical_member = WI_analytical[0, ..., i]

    # Plot analytical vs. data WI in the upper layer.
    plt.figure(1)
    plt.scatter(
        timesteps, WI_data_member, label=f"Layer {i} data", color=color, linestyle="-"
    )
    plt.plot(
        timesteps,
        WI_analytical_member,
        label=f"Layer {i} Peaceman",
        color=color,
        linestyle="--",
    )

    # Plot bhp predicted by Peaceman and data vs actual bhp in the upper layer.
    # NOTE: bhp predicted by data and actual bhp should be identical.
    bhp_data: np.ndarray = (
        injection_rate_per_second_per_cell_member / WI_data_member + pressures_member
    )
    bhp_analytical: np.ndarray = (
        injection_rate_per_second_per_cell_member / WI_analytical_member
        + pressures_member
    )
    plt.figure(2)
    plt.scatter(
        timesteps,
        bhp_data,
        label=rf"Layer {i}: calculated from data $WI$",
        color=color,
    )
    plt.plot(
        timesteps,
        bhp_analytical,
        label=rf"Layer {i}: calculated from Peaceman $WI$",
        color=color,
        linestyle="--",
    )
    plt.plot(
        timesteps,
        bhp_member,
        label=rf"Layer {i}: data",
        color=color,
        linestyle="-",
    )

plt.figure(1)
plt.legend()
plt.xlabel(r"$t\,[d]$")
plt.ylabel(r"$WI\,[m^4\cdot s/kg]$")
plt.title(r"$WI$")
plotting.save_fig_and_data(
    fig_1,
    pathlib.Path(ensemble_dirname) / "WI_data_vs_Peaceman.png",
)


plt.figure(2)
plt.legend()
plt.xlabel(r"$t\,[d]$")
plt.ylabel(r"$p\,[Pa]$")
plt.title(r"$p_{bhp}$ for various layers")
plotting.save_fig_and_data(
    fig_2, pathlib.Path(ensemble_dirname) / "pbh_data_vs_Peaceman.png"
)
plt.show()
