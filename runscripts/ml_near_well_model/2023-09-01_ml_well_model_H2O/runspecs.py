import numpy as np

# Matrix and fluid properties
PERMEABILITY: float = 101.32499658281449  # unit: mD
TEMPERATURE: float = 40  # unit: °C
# TEMPERATURE: float = 25  # unit: °C
INJECTION_RATE: float = 1e1  # unit: [m^3/d]

WELL_RADIUS: float = 0.25  # unit: [m]; Fixed during training.
RESERVOIR_SIZE: float = 5000  # unit: m;

SURFACE_DENSITY: float = 998.414  # unit: kg/m^3

# Ensemble specs
NPOINTS: int = 300  # number of ensemble members
NPRUNS: int = 5  # number of parallel runs

INIT_PRESSURE_MAX: float = 80
INIT_PRESSURE_MIN: float = 50
INIT_PRESSURES: np.ndarray = np.random.uniform(
    INIT_PRESSURE_MIN, INIT_PRESSURE_MAX, NPOINTS
)  # unit: bar
