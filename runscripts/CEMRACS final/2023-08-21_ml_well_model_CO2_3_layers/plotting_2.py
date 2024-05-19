"""Plot some nn and data things."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

import pyopmnearwell.utils.units as units

dirpath: str = os.path.dirname(__file__)
savepath: str = os.path.join(dirpath, "test_1")
os.makedirs(savepath, exist_ok=True)

with open(os.path.join(savepath, "Bottom_Hole_Pressure.ascii")) as f:
    lines: list[str] = f.readlines()[5:244]
    bhps_nn: np.ndarray = np.array([float(line.split("\t")[1]) for line in lines])
    bhps_Peaceman: np.ndarray = np.array([float(line.split("\t")[2]) for line in lines])
    pressures_125: np.ndarray = np.array(
        [float(line.split("\t")[-1]) for line in lines]
    )
    bhps_25: np.ndarray = np.array([float(line.split("\t")[3]) for line in lines])
    pressures_25: np.ndarray = np.array([float(line.split("\t")[-3]) for line in lines])
    bhps_50: np.ndarray = np.array([float(line.split("\t")[4]) for line in lines])
    pressures_50: np.ndarray = np.array([float(line.split("\t")[-2]) for line in lines])

times = np.linspace(1, 240, 239)

fig, ax = plt.subplots()
ax.plot(
    times,
    bhps_nn - pressures_125,
    label="125x125m nn",
)
# ax.plot(
#     times,
#     bhps_Peaceman - pressures_125,
#     label="125x125m Peaceman",
# )
ax.plot(
    times,
    bhps_25 - pressures_125,
    label="25x25m",
)
ax.plot(
    times,
    bhps_50 - pressures_125,
    label="50x50m",
)
ax.set_xlabel(r"$t\,[h]$")
ax.set_ylabel(r"$\Delta p\,[bar]$")
ax.set
ax.set_yticks(
    np.linspace((bhps_nn - pressures_125).min(), (bhps_50 - pressures_125).max(), 20)
)

fig.legend()
fig.tight_layout()
plt.savefig(os.path.join(savepath, f"delta_p.png"))
