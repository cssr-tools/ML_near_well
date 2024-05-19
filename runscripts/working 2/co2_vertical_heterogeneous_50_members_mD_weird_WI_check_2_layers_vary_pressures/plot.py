import pathlib

import matplotlib.pyplot as plt
import numpy as np
from pyopmnearwell.utils import units

dirname: pathlib.Path = pathlib.Path(__file__).parent

for filename in dirname.iterdir():
    if filename.suffix == ".ascii":
        with filename.open("r") as f:
            lol: list[str] = f.readlines()[3:124]
            averaged_values: list[float] = [
                sum(map(float, line.split("\t")[1:])) for line in lol
            ]
            if filename.stem == "layer_1_p_10":
                cell_pressure: np.ndarray = np.array(averaged_values)
            elif filename.stem == "layer_1_p_w":
                well_pressure: np.ndarray = np.array(averaged_values)
            elif filename.stem == "layer_1_q":
                injection_rate: np.ndarray = np.array(averaged_values)

WI: np.ndarray = injection_rate / (
    (well_pressure - cell_pressure) * units.BAR_TO_PASCAL
)
timesteps: np.ndarray = np.linspace(0, 5, len(WI))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# ax3 = ax1.twinx()

ax1.plot(timesteps, WI, label=r"WI")
ax1.set_xlabel(r"time $[d]$")
ax1.set_ylabel(r"$[m^4 \cdot s/kg]$")

ax2.plot(timesteps, well_pressure, label=r"$p_w$")
ax2.plot(timesteps, cell_pressure, label=r"$p_i$")
ax2.set_ylabel(r"$[Pa]$")

# ax3.plot(timesteps, injection_rate, label=r"$q$")
# ax3.set_ylabel(r"$[m/s]$")

fig.legend()

plt.show()
plt.savefig(dirname / "result.png", bbox_inches="tight")
