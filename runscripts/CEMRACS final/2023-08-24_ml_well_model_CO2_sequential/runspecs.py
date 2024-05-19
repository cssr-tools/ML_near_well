import math

import numpy as np

import pyopmnearwell.utils.units as units

# File names etc
ENSEMBLE_DIR: str = "ensemble_runs"
MLMODEL_DIR: str = "ml_model"

# Matrix and fluid properties
INJECTION_RATE: float = 5.352087e3  # unit: [m^3/d]

X: float = 2.500000e-01  # Outer coordinates of first cell.
Y: float = -1.443376e-01
WELL_RADIUS: float = math.sqrt(X**2 + Y**2)  # unit: [m]; Fixed during training.
RESERVOIR_SIZE: float = 5000  # unit: m;

SURFACE_DENSITY: float = 1.86843
DENSITY: float = 172.605  # unit: kg/m^3; for 57.5 bar, 25 °C.
VISCOSITY: float = 1.77645e-05  # unit: Pa*s; for 57.5 bar, 25 °C

# Ensemble specs
NPOINTS: int = 500
NPOINTS_PER_VALUE: tuple[int, int, int] = (30, 30, 30, 30)
NPRUNS: int = 5  # Number of parallel runs.

INIT_PRESSURE_MAX: float = 90
INIT_PRESSURE_MIN: float = 50
INIT_PRESSURES: np.ndarray = np.random.uniform(
    INIT_PRESSURE_MIN, INIT_PRESSURE_MAX, NPOINTS_PER_VALUE[0]
)  # unit: bar

INIT_TEMPERATURE_MAX: float = 20
INIT_TEMPERATURE_MIN: float = 40
INIT_TEMPERATURES: np.ndarray = np.random.uniform(
    INIT_TEMPERATURE_MIN, INIT_TEMPERATURE_MAX, NPOINTS_PER_VALUE[1]
)  # unit: °C

PERMEABILITY_MAX: float = 1e-11 * units.M2_TO_MILIDARCY
PERMEABILITY_MIN: float = 1e-13 * units.M2_TO_MILIDARCY
PERMEABILITIES: np.ndarray = np.random.uniform(
    PERMEABILITY_MIN, PERMEABILITY_MAX, NPOINTS_PER_VALUE[2]
)  # unit: mD

INJECTION_RATE_MIN: float = 0
INJECTION_RATE_MAX: float = 0
INJECTION_RATES: np.ndarray = np.array(
    [
        np.random.uniform(INJECTION_RATE_MIN, INJECTION_RATE_MAX, size=20)
        for _ in range(NPOINTS_PER_VALUE[3])
    ]
)  # unit: m^3/day

INJECTION_TIME_MIN: int = 1
INJECTION_TIME_MAX: float = 10
INJECTION_TIMES: np.ndarray = np.array(
    [
        np.random.random_integers(INJECTION_TIME_MIN, INJECTION_TIME_MAX, size=20)
        for _ in range(NPOINTS_PER_VALUE[3])
    ]
)  # unit: days
