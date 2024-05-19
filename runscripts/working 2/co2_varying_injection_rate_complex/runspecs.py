import pathlib
import random
from typing import Any

from pyopmnearwell.utils import formulas, units

dirname: pathlib.Path = pathlib.Path(__file__).parent

OPM_ML: str = "/home/peter/Documents/2023_CEMRACS/opm_ml"
FLOW_ML: str = f"{OPM_ML}/build/opm-simulators/bin/flow_gaswater_dissolution_diffuse"
OPM: str = "/home/peter/Documents/2023_CEMRACS/opm"
FLOW: str = f"{OPM}/build/opm-simulators/bin/flow"


# Fixed values for all runs
num_layers: int = 5
reservoir_height: float = 20.0  # unit: [m]
well_radius: float = 0.8  # unit: [m]
porosity: float = 0.36  # unit: [-]

surface_density: float = 1.86843  # unit: [kg/m^3]
init_temperature: float = 25  # unit: [°C]

##########
# Ensemble
##########
npoints: int = 50
num_zcells: int = num_layers * 5
num_xcells: int = 80


variables: dict[str, Any] = {
    f"inj_rate_{i}": (5e5, 5e6, npoints) for i in range(3)
}  # unit: [kg/d]

variables.update(
    {
        "INIT_PRESSURE": (
            50 * units.BAR_TO_PASCAL,
            150 * units.BAR_TO_PASCAL,
            npoints,
        )  # unit: [Pa]
    }
)

constants: dict[str, Any] = {
    "INIT_TEMPERATURE": init_temperature,  # unit: [°C])
    "SURFACE_DENSITY": surface_density,  # unit: [kg/m^3]
    # NOTE: 1e2 m^3/day is approx 2L/s for each meter of well.
    "WELL_RADIUS": well_radius,  # unit: [m]; Fixed during training.
    "POROSITY": porosity,  # unit [-]
    "PERM": 2000,  # unit: [mD]
    #
    "NUM_LAYERS": num_layers,
    "NUM_ZCELLS": num_zcells,
    "NUM_XCELLS": num_xcells,
    "LENGTH": 250,
    "HEIGHT": reservoir_height,
    "FLOW": FLOW,
    "OPM": OPM,
}

runspecs_ensemble: dict[str, Any] = {
    "name": "ensemble_1",
    "npoints": npoints,  # number of ensemble members
    "npruns": min(npoints, 5),  # number of parallel runs
    "variables": variables,
    "constants": constants,
}

##########
# Training
##########
trainspecs_1: dict[str, Any] = {
    "name": "trainspecs_1",
    # Data conversion/padding
    "pressure_unit": "Pascal",
    "permeability_log": False,
    "WI_log": False,
    "pressure_padding": "neighbor",
    "saturation_padding": "zeros",
    "permeability_padding": "zeros",
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "architecture": "fcnn",
    "features": [
        "pressure_upper",
        "pressure",
        "pressure_lower",
        "saturation_upper",
        "saturation",
        "saturation_lower",
        "injection_rate_upper",
        "injection_rate",
        "injection_rate_lower",
        "radius",
        "total_injected_volume",
        "PI_analytical",
    ],
    #
    "kerasify": True,
}


trainspecs_2: dict[str, Any] = {
    "name": "trainspecs_2",
    # Data conversion/padding
    "pressure_unit": "Pascal",
    "permeability_log": False,
    "WI_log": False,
    "pressure_padding": "neighbor",
    "saturation_padding": "zeros",
    "permeability_padding": "zeros",
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "architecture": "rnn",
    "features": [
        "pressure_upper",
        "pressure",
        "pressure_lower",
        "saturation_upper",
        "saturation",
        "saturation_lower",
        "injection_rate_upper",
        "injection_rate",
        "injection_rate_lower",
        "radius",
        "total_injected_volume",
        "PI_analytical",
    ],
    #
    "kerasify": False,
}

trainspecs_3: dict[str, Any] = {
    "name": "trainspecs_3",
    # Data conversion/padding
    "pressure_unit": "Pascal",
    "permeability_log": False,
    "WI_log": False,
    "pressure_padding": "neighbor",
    "saturation_padding": "zeros",
    "permeability_padding": "zeros",
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "architecture": "lstm",
    "features": [
        "pressure_upper",
        "pressure",
        "pressure_lower",
        "saturation_upper",
        "saturation",
        "saturation_lower",
        "injection_rate_upper",
        "injection_rate",
        "injection_rate_lower",
        "radius",
        "total_injected_volume",
        "PI_analytical",
    ],
    #
    "kerasify": False,
}

trainspecs_4: dict[str, Any] = {
    "name": "trainspecs_4",
    # Data conversion/padding
    "pressure_unit": "Pascal",
    "permeability_log": False,
    "WI_log": False,
    "pressure_padding": "neighbor",
    "saturation_padding": "zeros",
    "permeability_padding": "zeros",
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "architecture": "gru",
    "features": [
        "pressure_upper",
        "pressure",
        "pressure_lower",
        "saturation_upper",
        "saturation",
        "saturation_lower",
        "injection_rate_upper",
        "injection_rate",
        "injection_rate_lower",
        "radius",
        "total_injected_volume",
        "PI_analytical",
    ],
    #
    "kerasify": False,
}


##########
# Integration
##########
constants_1: dict[str, Any] = {
    f"PERM_{i}": random.uniform(8e-13, 2e-12) * units.M2_TO_MILIDARCY
    for i in range(num_layers)
}
constants_1.update(
    {
        "INIT_PRESSURE": 65 * units.BAR_TO_PASCAL,  # unit: [Pa]
        "INIT_TEMPERATURE": init_temperature,  # unit: [°C]
        #
        # "INJECTION_RATE": injection_rate * surface_density,  # unit: [kg/d]
        # "INJECTION_TIME": injection_time,  # unit: [day]
        "WELL_RADIUS": well_radius,  # unit: [m]; Fixed during training.
        "POROSITY": porosity,  # unit [-]
        #
        "RESERVOIR_SIZE": 1100,  # unit: [m]
        "HEIGHT": reservoir_height,  # unit: [m]
        "NUM_LAYERS": num_layers,
        "OPM_ML": OPM_ML,
        "FLOW_ML": FLOW_ML,
    }
)
runspecs_integration_1: dict[str, Any] = {
    "name": "integration_1",
    "ensemble_name": "ensemble_2",
    "nn_name": "trainspecs_1",
    "variables": {
        "GRID_SIZE": [68],  # , 5 , 10, 25, 50, 100, 5, 10, 25, 50, 100],
        "RUN_NAME": [
            "8x8m_Peaceman_more_zcells",
            # "100x100m_NN",
            # "50x50m_NN",
            # "21x21m_NN",
            # "11x11m_NN",
            # "5.5x5.5m_NN",
            # "100x100m_Peaceman",
            # "50x50m_Peaceman",
            # "21x21m_Peaceman",
            # "11x11m_Peaceman",
            # "5.5x5.5m_Peaceman",
        ],
        "NUM_ZCELLS": [num_layers * 5],  # + [num_layers] * 10,
    },
    "constants": constants_1,
    "PI_unit": "m^2",
}

# Taken from a fine-scale run with significant vertical flow.
# \co2_vertical_heterogeneous_50_members_mD_weird_WI_check\ensemble_2\runfiles_2
constants_2: dict[str, Any] = {
    "PERM_0": 751,
    "PERM_1": 4099,
    "PERM_2": 571,
    "PERM_3": 3784,
    "PERM_4": 2197,
}
constants_2.update(
    {
        "INIT_PRESSURE": 65 * units.BAR_TO_PASCAL,  # unit: [Pa]
        "INIT_TEMPERATURE": init_temperature,  # unit: [°C]
        #
        # "INJECTION_RATE": injection_rate * surface_density,  # unit: [kg/d]
        # "INJECTION_TIME": injection_time,  # unit: [day]
        "WELL_RADIUS": well_radius,  # unit: [m]; Fixed during training.
        "POROSITY": porosity,  # unit [-]
        #
        "RESERVOIR_SIZE": 1100,  # unit: [m]
        "HEIGHT": reservoir_height,  # unit: [m]
        "NUM_LAYERS": num_layers,
        "OPM_ML": OPM_ML,
        "FLOW_ML": FLOW_ML,
    }
)
runspecs_integration_2: dict[str, Any] = {
    "name": "integration_2",
    "ensemble_name": "ensemble_2",
    "nn_name": "trainspecs_1",
    "variables": {
        "GRID_SIZE": [68],  # 5, 10, 25, 50, 100],  # , 5, 10, 25, 50, 100],
        "RUN_NAME": [
            "8x8m_Peaceman_more_zcells",
            # "100x100m_NN",
            # "50x50m_NN",
            # "21x21m_NN",
            # "11x11m_NN",
            # "5.5x5.5m_NN",
            # "100x100m_Peaceman",
            # "50x50m_Peaceman",
            # "21x21m_Peaceman",
            # "11x11m_Peaceman",
            # "5.5x5.5m_Peaceman",
        ],
        "NUM_ZCELLS": [num_layers * 5],  # + [num_layers] * 10,
    },
    "constants": constants_2,
    "PI_unit": "m^2",
}
