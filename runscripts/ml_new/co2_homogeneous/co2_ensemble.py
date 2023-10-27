import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pyopmnearwell.ml import ensemble
from pyopmnearwell.utils import formulas, units

dirname: str = os.path.dirname(__file__)
os.makedirs(os.path.join(dirname, "ensemble"), exist_ok=True)
os.makedirs(os.path.join(dirname, "dataset"), exist_ok=True)
os.makedirs(os.path.join(dirname, "nn"), exist_ok=True)
os.makedirs(os.path.join(dirname, "integration"), exist_ok=True)
OPM: str = "/home/peter/Documents/2023_CEMRACS/opm"
FLOW: str = f"{OPM}/build/opm-simulators/bin/flow"
OPM_ML: str = "/home/peter/Documents/2023_CEMRACS/opm_ml"
FLOW_ML: str = f"{OPM_ML}/build/opm-simulators/bin/flow_gaswater_dissolution_diffuse"

# Run ensemble
height: float = 50
num_zcells: int = 10
npoints: int = 5

variables: dict[str, tuple[float, float, int]] = {
    "INIT_PRESSURE": (
        50 * units.BAR_TO_PASCAL,
        130 * units.BAR_TO_PASCAL,
        npoints,
    ),  # unit: [Pa]
}


runspecs_ensemble: dict[str, Any] = {
    "npoints": npoints,  # number of ensemble members
    "npruns": 5,  # number of parallel runs
    "variables": variables,
    "constants": {
        "PERMX": 1e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERMZ": 1e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "INIT_TEMPERATURE": 25,  # unit: [°C])
        "SURFACE_DENSITY": 1.86843,  # unit: [kg/m^3]
        "INJECTION_RATE": 3e5 * 1.86843,  # unit: [kg/d]
        "INJECTION_RATE_PER_SECOND": 3e5
        * units.Q_per_day_to_Q_per_seconds,  # unit: [m^3/s]
        "WELL_RADIUS": 0.25,  # unit: [m]; Fixed during training.
        "FLOW": FLOW,
        "NUM_ZCELLS": num_zcells,
        "HEIGHT": height,
    },
}

co2_ensemble = ensemble.create_ensemble(
    runspecs_ensemble,
)
ensemble.setup_ensemble(
    os.path.join(dirname, "ensemble"),
    co2_ensemble,
    os.path.join(dirname, "co2_ensemble.mako"),
    recalc_grid=False,
    recalc_sections=True,
    recalc_tables=False,
)
data: dict[str, Any] = ensemble.run_ensemble(
    FLOW,
    os.path.join(dirname, "ensemble"),
    runspecs_ensemble,
    ecl_keywords=["PRESSURE", "SGAS"],
    init_keywords=[],
    summary_keywords=[],
    num_report_steps=100,
)


# Get dataset

# Take only every 3rd time step, do not take the initial time step.
features: np.ndarray = np.array(
    ensemble.extract_features(
        data,
        keywords=["PRESSURE", "SGAS"],
        keyword_scalings={"PRESSURE": units.BAR_TO_PASCAL},
    )
)[..., 1::3, :, :]

# Cut off pressure at well cells and pore volume cells. Average pressure and saturation
# per z cell layer.
pressures: np.ndarray = np.average(
    features[..., 0].reshape(features.shape[0], features.shape[1], num_zcells, -1)[
        ..., 1:-2
    ],
    axis=-1,
)  # ``shape = (npoints, num_timesteps/3, num_layers)
saturations: np.ndarray = np.average(
    features[..., 1].reshape(features.shape[0], features.shape[1], num_zcells, -1)[
        ..., 1:-2
    ],
    axis=-1,
)  # ``shape = (npoints, num_timesteps/3, num_layers)

timesteps: np.ndarray = np.arange(features.shape[-3])  # No unit.

# Calculate WI manually (from the averaged pressures instead of an equivalent well
# radius).
bhps: np.ndarray = features[..., 0].reshape(
    features.shape[0], features.shape[1], num_zcells, -1
)[
    ..., 0
]  # ``shape = (npoints, num_timesteps/3, num_layers)

WI: np.ndarray = runspecs_ensemble["constants"]["INJECTION_RATE_PER_SECOND"] / (
    bhps - pressures
)

# Features are, in the following order:
# 1. PRESSURE - cell
# 2. SGAS - cell
# 3. TIME
# shape ``shape = (npoints, num_timesteps/3, num_layers, 3)``

ensemble.store_dataset(
    np.stack(
        [
            pressures,
            saturations,
            np.broadcast_to(timesteps[..., None], pressures.shape),
        ],
        axis=-1,
    ),
    WI[..., None],
    os.path.join(dirname, "dataset_averaged_pressure"),
)

for pressures_member, saturations_member, target, i in list(
    zip(
        pressures[::npoints, ..., 0],
        saturations[::npoints, ..., 0],
        WI[::npoints, :, 0],
        range(pressures[::npoints, ...].shape[0]),
    )
):
    # Loop through all time steps and collect analytical WIs.
    WI_analytical: list[float] = []
    for j in range(pressures_member.shape[0]):
        pressure = pressures_member[j]
        saturation = saturations_member[j]
        # Evalute density and viscosity.
        densities: list[float] = []
        viscosities: list[float] = []
        for phase in ["CO2", "water"]:
            densities.append(
                formulas.co2brinepvt(
                    pressure=pressure,
                    temperature=runspecs_ensemble["constants"]["INIT_TEMPERATURE"]
                    + units.CELSIUS_TO_KELVIN,
                    property="density",
                    phase=phase,
                )
            )
            viscosities.append(
                formulas.co2brinepvt(
                    pressure=pressure,
                    temperature=runspecs_ensemble["constants"]["INIT_TEMPERATURE"]
                    + units.CELSIUS_TO_KELVIN,
                    property="viscosity",
                    phase=phase,
                )
            )
        # Calculate the well index from two-phase Peaceman. Note that the relative
        # permeabilty functions are quadratic. The analytical well index is in [m*s],
        # hence we need to devide by density to transform to [m^4*s/kg].
        WI_analytical.append(
            formulas.two_phase_peaceman_WI(
                k_h=runspecs_ensemble["constants"]["PERMX"]
                * units.MILIDARCY_TO_M2
                * height
                / num_zcells,
                r_e=formulas.equivalent_well_block_radius(200),
                r_w=0.25,
                rho_1=densities[0],
                mu_1=viscosities[0],
                k_r1=saturation**2,
                rho_2=densities[1],
                mu_2=viscosities[1],
                k_r2=(1 - saturation) ** 2,
            )
            / runspecs_ensemble["constants"]["SURFACE_DENSITY"]
        )
    # Compute total mobility
    plt.figure()
    plt.scatter(
        timesteps * 3 + 1,
        target,
        label="data",
    )
    plt.plot(
        timesteps * 3 + 1,
        WI_analytical,
        label="two-phase Peaceman",
    )
    plt.legend()
    plt.xlabel(r"$t\,[d]$")
    plt.ylabel(r"$WI\,[m^4\cdot s/kg]$")
    plt.savefig(os.path.join(dirname, "ensemble", f"WI_data_vs_Peaceman_{i}.png"))
    plt.show()

for pressures_member, saturations_member, target, i in list(
    zip(
        pressures[::npoints, ..., 0],
        saturations[::npoints, ..., 0],
        WI[::npoints, :, 0],
        range(pressures[::npoints, ...].shape[0]),
    )
):
    # Loop through all time steps and collect analytical WIs.
    WI_analytical = []
    for j in range(pressures_member.shape[0]):
        pressure = pressures_member[j]
        saturation = saturations_member[j]
        # Evalute density and viscosity.
        densities = []
        viscosities = []
        for phase in ["CO2", "water"]:
            densities.append(
                formulas.co2brinepvt(
                    pressure=pressure,
                    temperature=runspecs_ensemble["constants"]["INIT_TEMPERATURE"]
                    + units.CELSIUS_TO_KELVIN,
                    property="density",
                    phase=phase,
                )
            )
            viscosities.append(
                formulas.co2brinepvt(
                    pressure=pressure,
                    temperature=runspecs_ensemble["constants"]["INIT_TEMPERATURE"]
                    + units.CELSIUS_TO_KELVIN,
                    property="viscosity",
                    phase=phase,
                )
            )
        # Calculate the well index from two-phase Peaceman. Note that the relative
        # permeabilty functions are quadratic. The analytical well index is in [m*s],
        # hence we need to devide by density to transform to [m^4*s/kg].
        WI_analytical.append(
            formulas.two_phase_peaceman_WI(
                k_h=runspecs_ensemble["constants"]["PERMX"]
                * units.MILIDARCY_TO_M2
                * height
                / num_zcells,
                r_e=formulas.equivalent_well_block_radius(100),
                r_w=0.25,
                rho_1=densities[0],
                mu_1=viscosities[0],
                k_r1=saturation**2,
                rho_2=densities[1],
                mu_2=viscosities[1],
                k_r2=(1 - saturation) ** 2,
            )
            / runspecs_ensemble["constants"]["SURFACE_DENSITY"]
        )
    # Compute total mobility
    injection_rate: float = runspecs_ensemble["constants"]["INJECTION_RATE_PER_SECOND"]
    bhp_data: np.ndarray = injection_rate / target + pressure
    bhp_analytical: np.ndarray = injection_rate / np.array(WI_analytical) + pressure
    plt.figure()
    plt.scatter(
        timesteps * 3 + 1,
        bhp_data,
        label="data",
    )
    plt.plot(
        timesteps * 3 + 1,
        bhp_analytical,
        label="two-phase Peaceman",
    )
    plt.legend()
    plt.xlabel(r"$t\,[d]$")
    plt.ylabel(r"$p\,[Pa]$")
    plt.savefig(os.path.join(dirname, "ensemble", f"p_data_vs_Peaceman_{i}.png"))
    plt.show()


# ensemble.store_dataset(
#     np.stack(
#         [
#             pressures,
#             saturations,
#             features[..., 2],
#             np.broadcast_to(timesteps[..., None], features[..., 0].shape),
#         ],
#         axis=-1,
#     ),
#     WI[..., None],
#     os.path.join(dirname, "dataset_averaged_pressure"),
# )
