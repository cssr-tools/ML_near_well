""""Run nearwell CO2 storage simulations in OPM-Flow for an ensemble of varying
permeabilities, initial pressures and well radii.

Construct a dataset containing permeabilities, initial reservoir pressures, wellbore
radii and distance from well as features and well indices as targets.


Features:
    - permeability [m^2]
    - pressure [bar]
    - radius [m]

Targets:
    - WI [(m^3/s)/Pa]=[Nm/s]

"""
from __future__ import annotations

import logging
import math
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from ecl.eclfile.ecl_file import open_ecl_file
from mako.template import Template

from pyopmnearwell.utils.formulas import peaceman_WI

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

# Create ensemble members.
npoints, npruns = (20, 20, 10), 5
permeabilities: np.ndarray = (
    np.random.uniform(1e-13, 1e-11, npoints[0])[..., None] * M2_TO_MILIDARCY
)  # unit: [mD], ``shape=(npoints[0],1)``
init_pressures: np.ndarray = (
    np.random.uniform(7e6, 1e7, npoints[1])[..., None] * PASCAL_TO_BAR
)  # unit: [bar]. ``shape=(npoints[1],1)``
well_radii: np.ndarray = np.random.uniform(0.01, 0.1, npoints[2])[
    ..., None
]  # unit: [m], ``shape=(npoints[2],1)``

permeabilities_v, init_pressures_v, well_radii_v = np.meshgrid(
    permeabilities.flatten(),
    init_pressures.flatten(),
    well_radii.flatten(),
    indexing="ij",
)

ensemble = np.stack((permeabilities_v.flatten(), init_pressures_v.flatten()), axis=-1)

# Set flow runspecs.
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
for i, (permeability, init_pressure, well_radius) in enumerate(ensemble):
    var = {
        "permeability_x": permeability,
        "init_pressure": init_pressure,
        "well_radius": well_radius,
    }
    filledtemplate = mytemplate.render(**var)
    with open(
        os.path.join(ensemble_path, f"RESERVOIR{i}.DATA"),
        "w",
        encoding="utf8",
    ) as file:
        file.write(filledtemplate)  # type: ignore

# Store final grid cell pressures for each ensemble member.
pressure_lst: list[np.ndarray] = []

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
            pressure_lst.append(np.array(ecl_file.iget_kw("PRESSURE"))[-1])

        # Remove the run files and result folder (except for the first one).
        if npruns * i + j > 0:
            shutil.rmtree(
                os.path.join(ensemble_path, f"results{npruns*i+j}"),
            )
            os.remove(os.path.join(ensemble_path, f"RESERVOIR{npruns*i+j}.DATA"))
os.remove(os.path.join(ensemble_path, f"TABLES.INC"))
os.remove(os.path.join(ensemble_path, f"CAKE.INC"))


# Calculate cell center distance from well.
with open(os.path.join(dirpath, "CAKE.INC"), "r") as radii_file:
    # Calculate the radius of the midpoint of each cell
    lines: list[str] = radii_file.readlines()[9:410]
    assert len(lines) == 401
    distances: np.ndarray = np.array(
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
    inner_radii: np.ndarray = distances[:-1]
    outer_radii: np.ndarray = distances[1:]
    distances = (inner_radii + outer_radii) / 2  # unit: [m]

# Truncate the well cells and cells in the close proximity, as well as the outermost
# cell (pore-volume) to sanitize the dataset (WI close to cell behaves weirdly).
distances = distances[4:-1]
assert distances.shape == (395,)

permeabilities_v, init_pressures_v, well_radii_v, radii_v = np.meshgrid(
    permeabilities.flatten(),
    init_pressures.flatten(),
    well_radii,
    distances,
    indexing="ij",
)

# Features are in the following order
# 1. permeability
# 2. initial reservoir pressure
# 3. well radius
# 3. distance cell center from well
# Furthermore, first the permeabilities are cycled through, then the initial pressures,
# then the well radii, then the distances from well (in four nested cycles, where the
# permeabilites are on the lowest cycle and distances on the highest).
features = np.stack(
    [
        permeabilities_v.flatten(),
        init_pressures_v.flatten(),
        well_radii_v.flatten(),
        radii_v.flatten(),
    ],
    axis=-1,
)  # ``shape=(npoints[0] * npoints[1] * npoints[2] * 395, 3)``

# Calculate the well indices from injection rate, cell pressures and bhp.
INJECTION_RATE: float = 5.352087e3 * Q_per_day_to_Q_per_seconds  # unit: [m^3/s]
pressures: np.ndarray = (
    np.array(pressure_lst)[..., None] * BAR_TO_PASCAL
)  # unit: [Pa], ``shape=(npoints[0] * npoints[1] * npoints[2], 400, 1)``,

# The bottom hole pressure is taken at the innermost cell.
bhp: np.ndarray = pressures[:, 0]

# Calculate well indices and truncate.
WI_t: np.ndarray = (INJECTION_RATE / (bhp[..., None] - pressures))[
    :, 4:-1
]  # ``shape=(npoints[0] * npoints[1], 395, 1)``

# Store the features (permeabilities, pressures, radii and distances from well) and
# targets (WI) as a dataset.
ds = tf.data.Dataset.from_tensor_slices((features, WI_t.flatten()[..., None]))
ds.save(os.path.join(ensemble_path, "dataset_WI"))

# Fix pressure and well radius, plot multiple permeabilities; compare to Peaceman well
# model.
features, targets = next(iter(ds.batch(batch_size=len(ds)).as_numpy_iterator()))
features_varying_perm = features.reshape((npoints[0], npoints[1], npoints[2], 395, 3))[
    ::5, 0, 0, ...
]
targets_varying_perm = targets.reshape((npoints[0], npoints[1], npoints[2], 395, 1))[
    ::5, 0, 0, ...
]
peaceman_varying_perm = [
    peaceman_WI(K, r_e, r_w) for K, r_e, r_w in features_varying_perm
]
for feature, target, peaceman in zip(
    features_varying_perm, targets_varying_perm, peaceman_varying_perm
):
    plt.scatter(
        feature[..., 3].flatten(), target.flatten(), label=rf"$k={feature[0][0]}$ [m^2]"
    )
    plt.plot(
        feature[..., 3].flatten(),
        peaceman,
        label=rf"$k={feature[0][0]}$ [m^2] Peaceman",
    )

plt.legend()
plt.xlabel(r"$r$ [m]")
plt.ylabel(r"$WI$ [Nm/s]")
plt.title(
    rf"initial reservoir pressure ${feature[0][1]}$ [bar],"
    + rf" well radius ${feature[0][2]}$ [m]"
)
plt.savefig(os.path.join(dirpath, run_name, "k_r_to_WI.png"))
plt.show()

# Fix pressure and permeability, plot multiple well radii; compare to Peaceman well
# model.
features, targets = next(iter(ds.batch(batch_size=len(ds)).as_numpy_iterator()))
features_varying_well_radius = features.reshape(
    (npoints[0], npoints[1], npoints[2], 395, 3)
)[0, 0, ::2, ...]
targets_varying_well_radius = targets.reshape(
    (npoints[0], npoints[1], npoints[2], 395, 1)
)[0, 0, ::2, ...]
peaceman_varying_well_radius = [
    peaceman_WI(K, r_e, r_w) for K, r_e, r_w in features_varying_well_radius
]
for feature, target, peaceman in zip(
    features_varying_well_radius,
    targets_varying_well_radius,
    peaceman_varying_well_radius,
):
    plt.scatter(
        feature[..., 3].flatten(),
        target.flatten(),
        label=rf"$well radius={feature[0][2]}$ [m^2]",
    )
    plt.plot(
        feature[..., 3].flatten(),
        peaceman,
        label=rf"$well radius={feature[0][2]}$ [m^2] Peaceman",
    )
    logger.info(feature)
plt.legend()
plt.xlabel(r"$r$ [m]")
plt.ylabel(r"$WI$ [Nm/s]")
plt.title(
    rf"permeability ${feature}$ [m], initial reservoir pressure ${feature[0][1]}$ [bar]"
)
plt.savefig(os.path.join(dirpath, run_name, "well_radius_d_to_WI.png"))
plt.show()


# Three dimensional plot of permeabilities and distance from well vs. WI, pressure and
# well radius fixed.
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(
    permeabilities_v[:, 0, ...],
    radii_v[:, 0, ...],
    np.squeeze(WI_t[:: npoints[1]]),
    rstride=1,
    cstride=1,
    cmap="viridis",
    edgecolor="none",
)
ax.set_xlabel(r"$k$ [m^2]")
ax.set_ylabel(r"$r$ [m]")
ax.set_title(r"$WI$ [Nm/s]")
plt.title(rf"initial reservoir pressure ${feature[0][1]}$ [bar], well radius $$ [m]")
plt.savefig(os.path.join(dirpath, run_name, "k_r_to_WI_3d.png"))
plt.show()
