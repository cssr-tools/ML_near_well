import math

import pyopmnearwell.utils.units as units

# Matrix and fluid properties
PERMEABILITY: float = 1e-12 * units.M2_TO_MILIDARCY  # unit: mD
TEMPERATURE: float = 30.9780  # unit: °C
INJECTION_RATE: float = 5.352087e3  # unit: [kg/d]
X: float = 2.500000e-01  # Outer coordinates of first cell.
Y: float = -1.443376e-01
WELL_RADIUS: float = math.sqrt(X**2 + Y**2)  # unit: [m]; Fixed during training.
DENSITY: float = 12.9788  # unit: kg/m^3; for 72 bar, 30.9780 °C. Is this at surface conditions or not?
VISCOSITY: float = 1.52786e-05  # unit: Pa*s; for 72 bar, 30.9780 °C

# Ensemble specs
NPOINTS: int = 500  # number of ensemble members
NPRUNS: int = 5  # number of parallel runs
