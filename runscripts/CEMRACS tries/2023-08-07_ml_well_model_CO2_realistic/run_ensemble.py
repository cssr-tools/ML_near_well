""""Run nearwell CO2 storage simulations in OPM-Flow for an ensemble of varying
permeabilities and initial pressures.

Construct a dataset containing permeabilities, initial reservoir pressures and radii as
features and well indices as targets.

Features:
    1. permeability [mD]
    2. pressure [bar]
    3. radius [m]

Targets:
    1. WI [m*s]

"""
from __future__ import annotations

import logging
import math
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from ecl.eclfile.ecl_file import EclFile, open_ecl_file
from ecl.summary import EclSum
from mako.template import Template

# Unit conversions.
M2_TO_MILIDARCY = 1.01324997e15
PASCAL_TO_BAR = 1.0e-5
BAR_TO_PASCAL = 1e5
Q_per_day_to_Q_per_seconds = 1.0 / 86400


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


run_name: str = "ensemble_runs"

dirpath: str = os.path.dirname(os.path.realpath(__file__))
os.makedirs(os.path.join(dirpath, run_name), exist_ok=True)
ensemble_path = os.path.join(dirpath, run_name)

shutil.copyfile(
    os.path.join(dirpath, "TABLES.INC"),
    os.path.join(dirpath, run_name, "TABLES.INC"),
)
shutil.copyfile(
    os.path.join(dirpath, "CAKE.INC"),
    os.path.join(dirpath, run_name, "CAKE.INC"),
)

# Set run specs for ensemble simulations.
# ``npoints`` is a tuple containing the number of ensemble members in each dimension
# (permeability and pressure).
npoints, npruns = (30, 20), 5
# Create permeability ensemble and convert to [mD].
permeabilities = (
    np.random.uniform(1e-13, 1e-11, npoints[0]) * M2_TO_MILIDARCY
)  # unit: mD
# Create permeability ensemble and convert to [bar^2].
init_pressures = np.random.uniform(7e6, 1e7, npoints[1]) * PASCAL_TO_BAR  # unit: bar

permeabilities_v, init_pressures_v = np.meshgrid(
    permeabilities, init_pressures, indexing="ij"
)
ensemble = np.stack((permeabilities_v.flatten(), init_pressures_v.flatten()), axis=-1)

FLOW = "/home/peter/Documents/2023_CEMRACS/opm/build/opm-simulators/bin/flow"
FLAGS = (
    " --linear-solver-reduction=1e-5 --relaxed-max-pv-fraction=0"
    + " --ecl-enable-drift-compensation=0 --newton-max-iterations=50"
    + " --newton-min-iterations=5 --tolerance-mb=1e-7 --tolerance-wells=1e-5"
    + " --relaxed-well-flow-tol=1e-5 --use-multisegment-well=false --enable-tuning=true"
    + " --enable-opm-rst-file=true --linear-solver=cprw"
    + " --enable-well-operability-check=false"
    + " --min-time-step-before-shutting-problematic-wells-in-days=1e-99"
)

# Create a deck file for each ensemble member
mytemplate = Template(filename=os.path.join(dirpath, "RESERVOIR.mako"))
for i, (permeability, init_pressure) in enumerate(ensemble):
    var = {"permeability_x": permeability, "init_pressure": init_pressure}
    filledtemplate = mytemplate.render(**var)
    with open(
        os.path.join(ensemble_path, f"RESERVOIR{i}.DATA"),
        "w",
        encoding="utf8",
    ) as file:
        file.write(filledtemplate)  # type: ignore

# Store final grid cell pressures for each ensemble member.
pressures: list[np.ndarray] = []

# Run OPM flow for each ensemble member.
for i in range(round(npoints[0] * npoints[1] / npruns)):
    os.system(
        f"{FLOW}"
        + f" {os.path.join(ensemble_path, f'RESERVOIR{npruns*i}.DATA')}"
        + f" --output-dir={os.path.join(ensemble_path, f'results{npruns*i}')} {FLAGS} & "
        f"{FLOW}"
        + f" {os.path.join(ensemble_path, f'RESERVOIR{npruns*i+1}.DATA')}"
        + f" --output-dir={os.path.join(ensemble_path, f'results{npruns*i+1}')} {FLAGS} & "
        f"{FLOW}"
        + f" {os.path.join(ensemble_path, f'RESERVOIR{npruns*i+2}.DATA')}"
        + f" --output-dir={os.path.join(ensemble_path, f'results{npruns*i+2}')} {FLAGS} & "
        f"{FLOW}"
        + f" {os.path.join(ensemble_path, f'RESERVOIR{npruns*i+3}.DATA')}"
        + f" --output-dir={os.path.join(ensemble_path, f'results{npruns*i+3}')} {FLAGS} & "
        f"{FLOW}"
        + f" {os.path.join(ensemble_path, f'RESERVOIR{npruns*i+4}.DATA')}"
        + f" --output-dir={os.path.join(ensemble_path, f'results{npruns*i+4}')} {FLAGS} & wait"
    )
    for j in range(npruns):
        with open_ecl_file(
            os.path.join(
                ensemble_path, f"results{npruns*i+j}", f"RESERVOIR{npruns*i+j}.UNRST"
            )
        ) as ecl_file:
            # The pressure array has shape ``[number_report_steps, number_cells]``
            pressures.append(np.array(ecl_file.iget_kw("PRESSURE"))[-1])

        # Remove the run files and result folder (except for the first one).
        if npruns * i + j > 0:
            shutil.rmtree(
                os.path.join(ensemble_path, f"results{npruns*i+j}"),
            )
            os.remove(os.path.join(ensemble_path, f"RESERVOIR{npruns*i+j}.DATA"))
os.remove(os.path.join(ensemble_path, f"TABLES.INC"))
os.remove(os.path.join(ensemble_path, f"CAKE.INC"))


# Create feature tensor.
permeabilities_t: np.ndarray = permeabilities[..., None]  # shape (npoints[0], 1)
init_pressures_t: np.ndarray = init_pressures[..., None]  # shape (npoints[1], 1)
# Calculate cell center radii.
with open(os.path.join(dirpath, "CAKE.INC"), "r") as radii_file:
    # Calculate the radius of the midpoint of each cell
    lines: list[str] = radii_file.readlines()[9:410]
    assert len(lines) == 401
    radii: np.ndarray = np.array(
        list(
            map(
                lambda x: math.sqrt(
                    float(x.strip("\n").split()[0]) ** 2
                    + float(x.strip("\n").split()[1]) ** 2
                ),
                lines,
            )
        )
    )
    inner_radii: np.ndarray = radii[:-1]
    outer_radii: np.ndarray = radii[1:]
    radii_t: np.ndarray = (inner_radii + outer_radii) / 2  # unit: [m]
# Truncate the well cells and cells in the close proximity, as well as the outermost
# cell (pore-volume) to sanitize the dataset (WI close to cell behaves weirdly).
radii_t = radii_t[4:-1]
assert radii_t.shape == (395,)

permeabilities_v, init_pressures_v, radii_v = np.meshgrid(
    permeabilities_t.flatten(), init_pressures_t.flatten(), radii_t, indexing="ij"
)
features = np.stack(
    [permeabilities_v.flatten(), init_pressures_v.flatten(), radii_v.flatten()], axis=-1
)  # ``shape=(npoints[0] * npoints[1] * 395, 3)``

# Calculate the well indices from injection rate, cell pressures and bhp.
INJECTION_RATE: float = 5.352087e3 * Q_per_day_to_Q_per_seconds  # unit: [kg/s]
pressures_t = (
    np.array(pressures)[..., None] * BAR_TO_PASCAL
)  # ``shape=(npoints[0] * npoints[1], 400, 1)``, unit: [Pa]
# Get the pressure value of the innermost cell, which equals the bottom hole pressure.
bhp_t: np.ndarray = pressures_t[:, 0]
# Truncate the cell pressures the same way the radii were truncated.
WI_t: np.ndarray = INJECTION_RATE / (
    bhp_t[..., None] - pressures_t[:, 4:-1]
)  # ``shape=(npoints[0] * npoints[1], 395, 1)``

# Store the features: permeabilities, pressures, radii, and targets: WI as a dataset.
# Features are in the following order
# 1. Permeability [mD]
# 2. initial reservoir pressure [bar]
# 3. cell radius [m]
# Furthermore, first the initial pressures and then the radii (in two nested loops,
# where the pressures are on the lower loop and radii on the higher).
# Targets are
# 1. WI [m*s]

ds = tf.data.Dataset.from_tensor_slices((features, WI_t.flatten()[..., None]))
ds.save(os.path.join(ensemble_path, "permeability_radius_WI"))

# Fix pressure, plot multiple permeabilities
features, targets = next(iter(ds.batch(batch_size=len(ds)).as_numpy_iterator()))
features_fixed_init_pressure = features.reshape((npoints[0], npoints[1], 395, 3))[
    ::1, 0, ...
]
targets_fixed_init_pressure = targets.reshape((npoints[0], npoints[1], 395, 1))[
    ::1, 0, ...
]
for feature, target in zip(features_fixed_init_pressure, targets_fixed_init_pressure):
    k = feature[0][0]
    plt.scatter(
        feature[..., 2].flatten()[::5], target.flatten()[::5], label=rf"$k={k}$ [m^2]"
    )
    logger.info(feature)
plt.legend()
plt.xlabel(r"$r$ [m]")
plt.ylabel(r"$WI$ [Nm/s]")
plt.title(rf"initial reservoir pressure ${feature[0][1]}$ [bar]")
plt.savefig(os.path.join(dirpath, run_name, "k_r_to_WI.png"))
plt.show()


# Three dimensional plot of permeabilities and radii vs. WI, pressure fixed.
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(
    permeabilities_v[:, 0, :],
    radii_v[:, 0, :],
    np.squeeze(WI_t[:: npoints[1]]),
    rstride=1,
    cstride=1,
    cmap="viridis",
    edgecolor="none",
)
ax.set_xlabel(r"$k$ [m^2]")
ax.set_ylabel(r"$r$ [m]")
ax.set_title(r"$WI$ [Nm/s]")

plt.title(rf"initial reservoir pressure ${feature[0][1]}$ [bar]")
plt.savefig(os.path.join(dirpath, run_name, "k_r_to_WI_3d.png"))
plt.show()
