import pathlib

import numpy as np
from pyopmnearwell.utils import formulas, units

OPM: pathlib.Path = (
    pathlib.Path("/home") / "peter" / "Documents" / "2023_CEMRACS" / "opm"
)
TEMPERATURE: float = 30
pressures: np.ndarray = np.linspace(20, 500, 10)

ratios: list[float] = []

for pressure in pressures:
    density: float = formulas.co2brinepvt(
        pressure * units.BAR_TO_PASCAL,
        TEMPERATURE + units.CELSIUS_TO_KELVIN,
        "density",
        "water",
        OPM,
    )
    viscosity: float = formulas.co2brinepvt(
        pressure * units.BAR_TO_PASCAL,
        TEMPERATURE + units.CELSIUS_TO_KELVIN,
        "viscosity",
        "water",
        OPM,
    )
    ratios.append(density / viscosity)
    print(
        f"At {pressure:.2f} bar: density {density:.2f}"
        + f" viscosity {viscosity:.10f} ratio {ratios[-1]:.2f}"
    )

ratio_min: float = min(ratios)
ratio_max: float = max(ratios)

print(
    f"min ration {ratio_min:.2f} max ratio {ratio_max:.2f}"
    + f" difference {(ratio_max / ratio_min) * 100 - 100:.2f} %"
)
