import copy
import random
from typing import Any

from pyopmnearwell.utils import formulas, units

OPM_ML: str = "/home/peter/Documents/2023_CEMRACS/opm_ml"
FLOW_ML: str = f"{OPM_ML}/build/opm-simulators/bin/flow_gaswater_dissolution_diffuse"
OPM: str = "/home/peter/Documents/2023_CEMRACS/opm"
FLOW: str = f"{OPM}/build/opm-simulators/bin/flow"

npoints: int = 500
npoints_3: int = 1000
npoints_4: int = 1000

num_zcells: int = 10
injection_rate: float = 1e4  # unit: [m^3/d]
surface_density: float = 998.414  # unit: [kg/m^3]


# NOTE: Vertical permeabilties are equal to horizontal permeability.
# variables_1: large differences in permeability.
variables_1: dict[str, tuple[float, float, int]] = {
    f"PERM_{i}": (  # unit: [mD]
        1e-15 * units.M2_TO_MILIDARCY,
        1e-12 * units.M2_TO_MILIDARCY,
        40,
    )
    for i in range(num_zcells)
}
variables_4: dict[str, tuple[float, float, int]] = copy.deepcopy(variables_1)

# variables_2: small differences in permeability.
variables_2: dict[str, tuple[float, float, int]] = {
    f"PERM_{i}": (  # unit: [mD]
        1e-14 * units.M2_TO_MILIDARCY,
        3e-13 * units.M2_TO_MILIDARCY,
        40,
    )
    for i in range(num_zcells)
}

# variables_3: even smaller differences in permeability.
variables_3: dict[str, tuple[float, float, int]] = {
    f"PERM_{i}": (  # unit: [mD]
        1e-14 * units.M2_TO_MILIDARCY,
        7.5e-14 * units.M2_TO_MILIDARCY,
        100,
    )
    for i in range(num_zcells)
}

for variables in [variables_1, variables_2, variables_3, variables_4]:
    variables.update(
        {
            "INIT_PRESSURE": (
                50 * units.BAR_TO_PASCAL,
                150 * units.BAR_TO_PASCAL,
                100,
            ),  # unit: [Pa]
        }
    )

constants: dict[str, Any] = {
    "INIT_TEMPERATURE": 25,  # unit: [°C])
    "SURFACE_DENSITY": surface_density,  # unit: [kg/m^3]
    "INJECTION_RATE": injection_rate * surface_density,  # unit: [kg/d]
    "INJECTION_RATE_PER_SECOND": injection_rate
    * units.Q_per_day_to_Q_per_seconds,  # unit: [m^3/s]
    # NOTE: 1e4 m^3/day is approx 200L/s for each meter of well. Setting this higher
    # (e.g., 6e4 m^3/day) will result in inaccurate results as the PVT values are
    # exceeded.
    "WELL_RADIUS": 0.25,  # unit: [m]; Fixed during training.
    "NUM_ZCELLS": num_zcells,
    "HEIGHT": 50,
    "FLOW": FLOW,
    "OPM": OPM,
}


runspecs_ensemble_1: dict[str, Any] = {
    "name": "ensemble_1",
    "npoints": npoints,  # number of ensemble members
    "npruns": min(npoints, 5),  # number of parallel runs
    "variables": variables_1,
    "constants": constants,
}
runspecs_ensemble_2: dict[str, Any] = {
    "name": "ensemble_2",
    "npoints": npoints,  # number of ensemble members
    "npruns": min(npoints, 5),  # number of parallel runs
    "variables": variables_1,
    "constants": constants,
}
runspecs_ensemble_3: dict[str, Any] = {
    "name": "ensemble_3",
    "npoints": npoints_3,  # number of ensemble members
    "npruns": min(npoints, 5),  # number of parallel runs
    "variables": variables_3,
    "constants": constants,
}

runspecs_ensemble_4: dict[str, Any] = {
    "name": "ensemble_4",
    "npoints": npoints_4,  # number of ensemble members
    "npruns": min(npoints, 5),  # number of parallel runs
    "variables": variables_1,
    "constants": constants,
}


# Integration
constants_integration_1: dict[str, Any] = copy.deepcopy(constants)
constants_integration_2: dict[str, Any] = copy.deepcopy(constants)

constants_integration_1.update(
    {
        f"PERM_{i}": random.uniform(  # unit: [mD]
            1e-14 * units.M2_TO_MILIDARCY,
            1e-10 * units.M2_TO_MILIDARCY,
        )
        for i in range(num_zcells)
    }
)
constants_integration_2.update(
    {
        f"PERM_{i}": random.uniform(  # unit: [mD]
            1e-14 * units.M2_TO_MILIDARCY,
            1e-13 * units.M2_TO_MILIDARCY,
        )
        for i in range(num_zcells)
    }
)

for constants_integration in [constants_integration_1, constants_integration_2]:
    constants_integration.update(
        {
            "INIT_PRESSURE": 55 * units.BAR_TO_PASCAL,  # unit: [Pa]
            "RESERVOIR_SIZE": 5000,  # unit: [m]
            "FLOW": FLOW_ML,
            "OPM": OPM_ML,
        }
    )

runspecs_integration_1: dict[str, Any] = {
    "variables": {
        "GRID_SIZE": [20, 20, 100],
        "RUN_NAME": ["125x125m_NN", "125x125m_Peaceman", "25x25m_Peaceman"],
    },
    "constants": constants_integration_1,
}
runspecs_integration_2: dict[str, Any] = {
    "variables": {
        "GRID_SIZE": [20, 20, 100],
        "RUN_NAME": ["125x125m_NN", "125x125m_Peaceman", "25x25m_Peaceman"],
    },
    "constants": constants_integration_2,
}

# Training
trainspecs_1: dict[str, Any] = {
    "name": "trainspecs_1",
    # Training hyperparameters
    "loss": "mse",  # mse, MeanAbsolutePercentageError, MeanSquaredLogarithmicError
    "epochs": 1000,  # int > 0
    # Data conversion/padding
    "pressure_unit": "bar",  # bar, Pascal
    "permeability_log": False,  # True, False
    "WI_log": False,  # True, False
    "pressure_padding": "zeros",  # zeros, init, neighbor
    "permeability_padding": "zeros",  # zeros,
    # Scaling/normalization
    "MinMax_scaling": True,  # True, False
    "Z-normalization": False,  # True, False
    # Network architecture
    "depth": 3,  # int > 0
    "hidden_dim": 10,  # int > 0
    "features": [
        "pressure_upper",
        "pressure",
        "pressure_lower",
        "permeability_upper",
        "permeability",
        "permeability_lower",
        "time",
    ],
}

trainspecs_2: dict[str, Any] = {
    "name": "trainspecs_2",
    # Training hyperparameters
    "loss": "mse",  # mse,
    "epochs": 1000,
    # Data conversion/padding
    "pressure_unit": "Pascal",  # bar, Pascal
    "permeability_log": True,  # True, False
    "WI_log": True,  # True, False
    "pressure_padding": "zeros",  # zeros, init, neighbor
    "permeability_padding": "zeros",  # zeros,
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "depth": 3,
    "hidden_dim": 10,
    "features": [
        "permeability",
    ],
}


trainspecs_3: dict[str, Any] = {
    "name": "trainspecs_3",
    # Training hyperparameters
    "loss": "MeanAbsolutePercentageError",
    "epochs": 1000,
    # Data conversion/padding
    "pressure_unit": "Pascal",  # bar, Pascal
    "permeability_log": True,  # True, False
    "WI_log": True,  # True, False
    "pressure_padding": "zeros",  # zeros, init, neighbor
    "permeability_padding": "zeros",  # zeros,
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "depth": 3,
    "hidden_dim": 10,
    "features": [
        "permeability",
    ],
}

trainspecs_4: dict[str, Any] = {
    "name": "trainspecs_4",
    # Training hyperparameters
    "loss": "MeanAbsolutePercentageError",
    "epochs": 1000,
    # Data conversion/padding
    "pressure_unit": "Pascal",  # bar, Pascal
    "permeability_log": True,  # True, False
    "WI_log": False,  # True, False
    "pressure_padding": "zeros",  # zeros, init, neighbor
    "permeability_padding": "zeros",  # zeros,
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "depth": 3,
    "hidden_dim": 10,
    "features": [
        "permeability",
    ],
}

trainspecs_5: dict[str, Any] = {
    "name": "trainspecs_5",
    # Training hyperparameters
    "loss": "MeanAbsolutePercentageError",
    "epochs": 1000,
    # Data conversion/padding
    "pressure_unit": "Pascal",  # bar, Pascal
    "permeability_log": False,  # True, False
    "WI_log": False,  # True, False
    "pressure_padding": "zeros",  # zeros, init, neighbor
    "permeability_padding": "zeros",  # zeros,
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "depth": 3,
    "hidden_dim": 10,
    "features": [
        "permeability",
    ],
}

trainspecs_6: dict[str, Any] = {
    "name": "trainspecs_6",
    # Training hyperparameters
    "loss": "MeanAbsolutePercentageError",  # mse, MeanAbsolutePercentageError
    "epochs": 1000,
    # Data conversion/padding
    "pressure_unit": "Pascal",  # bar, Pascal
    "permeability_log": False,  # True, False
    "WI_log": False,  # True, False
    "pressure_padding": "neighbor",  # zeros, init, neighbor
    "permeability_padding": "zeros",  # zeros,
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "depth": 3,
    "hidden_dim": 10,
    "features": [
        "pressure_upper",
        "pressure",
        "pressure_lower",
        "permeability_upper",
        "permeability",
        "permeability_lower",
        "time",
    ],
}

trainspecs_7: dict[str, Any] = {
    "name": "trainspecs_7",
    # Training hyperparameters
    "loss": "MeanSquaredLogarithmicError",  # mse, MeanAbsolutePercentageError
    "epochs": 1000,
    # Data conversion/padding
    "pressure_unit": "Pascal",  # bar, Pascal
    "permeability_log": False,  # True, False
    "WI_log": False,  # True, False
    "pressure_padding": "neighbor",  # zeros, init, neighbor
    "permeability_padding": "zeros",  # zeros,
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "depth": 3,
    "hidden_dim": 10,
    "features": [
        "pressure_upper",
        "pressure",
        "pressure_lower",
        "permeability_upper",
        "permeability",
        "permeability_lower",
        "time",
    ],
}

trainspecs_8: dict[str, Any] = {
    "name": "trainspecs_8",
    # Training hyperparameters
    "loss": "mse",  # mse,
    "epochs": 1000,
    # Data conversion/padding
    "pressure_unit": "Pascal",  # bar, Pascal
    "permeability_log": True,  # True, False
    "WI_log": True,  # True, False
    "pressure_padding": "neighbor",  # zeros, init, neighbor
    "permeability_padding": "zeros",  # zeros,
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "depth": 5,
    "hidden_dim": 10,
    "features": [
        "pressure_upper",
        "pressure",
        "pressure_lower",
        "permeability_upper",
        "permeability",
        "permeability_lower",
        "time",
    ],
}

trainspecs_9: dict[str, Any] = {
    "name": "trainspecs_9",
    # Training hyperparameters
    "loss": "mse",  # mse,
    "epochs": 1000,
    # Data conversion/padding
    "pressure_unit": "Pascal",  # bar, Pascal
    "permeability_log": False,  # True, False
    "WI_log": False,  # True, False
    "pressure_padding": "zeros",  # zeros, init, neighbor
    "permeability_padding": "zeros",  # zeros,
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "depth": 5,
    "hidden_dim": 10,
    "features": [
        "pressure",
        "permeability",
    ],
}

trainspecs_10: dict[str, Any] = {
    "name": "trainspecs_10",
    # Training hyperparameters
    "loss": "MeanAbsolutePercentageError",  # mse,
    "epochs": 1000,
    # Data conversion/padding
    "pressure_unit": "Pascal",  # bar, Pascal
    "permeability_log": False,  # True, False
    "WI_log": False,  # True, False
    "pressure_padding": "zeros",  # zeros, init, neighbor
    "permeability_padding": "zeros",  # zeros,
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "depth": 5,
    "hidden_dim": 10,
    "features": [
        "pressure",
        "permeability",
    ],
}

trainspecs_11: dict[str, Any] = {
    "name": "trainspecs_11",
    # Training hyperparameters
    "loss": "mse",  # mse,
    "epochs": 1000,
    "activation": "relu",
    # Data conversion/padding
    "pressure_unit": "Pascal",  # bar, Pascal
    "permeability_log": False,  # True, False
    "WI_log": False,  # True, False
    "pressure_padding": "zeros",  # zeros, init, neighbor
    "permeability_padding": "zeros",  # zeros,
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "depth": 5,
    "hidden_dim": 10,
    "features": ["pressure", "permeability", "time"],
    #
    "kerasify": False,
}


trainspecs_12: dict[str, Any] = {
    "name": "trainspecs_12",
    # Training hyperparameters
    "loss": "mse",  # mse,
    "epochs": 1000,
    "activation": "relu",
    # Data conversion/padding
    "pressure_unit": "Pascal",  # bar, Pascal
    "permeability_log": True,  # True, False
    "WI_log": True,  # True, False
    "pressure_padding": "zeros",  # zeros, init, neighbor
    "permeability_padding": "zeros",  # zeros,
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "depth": 5,
    "hidden_dim": 10,
    "features": ["pressure", "permeability", "time"],
    #
    "kerasify": False,
}

trainspecs_13: dict[str, Any] = {
    "name": "trainspecs_13",
    # Training hyperparameters
    "loss": "mse",  # mse,
    "epochs": 1000,
    "activation": "relu",
    # Data conversion/padding
    "pressure_unit": "Pascal",  # bar, Pascal
    "permeability_log": True,  # True, False
    "WI_log": True,  # True, False
    "pressure_padding": "zeros",  # zeros, init, neighbor
    "permeability_padding": "zeros",  # zeros,
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "depth": 5,
    "hidden_dim": 10,
    "features": [
        "pressure_upper",
        "pressure",
        "pressure_lower",
        "permeability_upper",
        "permeability",
        "permeability_lower",
        "time",
    ],
    #
    "kerasify": False,
}
