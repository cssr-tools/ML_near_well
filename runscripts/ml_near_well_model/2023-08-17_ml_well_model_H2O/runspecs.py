import math

import pyopmnearwell.utils.units as units

# Matrix and fluid properties
PERMEABILITY: float = 101.32499658281449  # unit: mD
TEMPERATURE: float = 40  # unit: °C
# TEMPERATURE: float = 25  # unit: °C
INJECTION_RATE: float = 2.401920e-01  # unit: [m^3/d]

X: float = 2.500000e-01  # Outer coordinates of first cell.
WELL_RADIUS: float = 0.25  # unit: [m]; Fixed during training.
RESERVOIR_SIZE: float = 5000  # unit: m;

SURFACE_DENSITY: float = 998.414  # unit: kg/m^3
DENSITY: float = 995.744  # unit: kg/m^3; for 100 bar, 40 °C.
VISCOSITY: float = 0.000654978  # 0.000881927  # unit: Pa*s; for 100 bar, 40 °C

# DENSITY: float = 1009.35  # unit: kg/m^3; for 30 bar, 25 °C.
# VISCOSITY: float = 0.000881927  # 0.000881927  # unit: Pa*s; for 30 bar, 25 °C

# Ensemble specs
NPOINTS: int = 300  # number of ensemble members
NPRUNS: int = 5  # number of parallel runs
