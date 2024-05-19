""""Run nearwell CO2 storage simulations in OPM-Flow for an ensemble of varying initial
initial pressures and construct a dataset containing cell pressures, time since
injection and cell center radii as features and well indices as targets.

Features:
    1. pressure [Pa]: Measured each report step at the cell centers.
    2. temperature [°C]: Measured at the beginning of the simulation.
    3. permeability [mD]: Fixed and homogeneous for the simulation.
    4. time [h]: Time since beginning of simulation at report step.
    5. radius [m]: Cell center radii.

Targets:
    1. WI [[m^4*s/kg]]: Calculated at the given radius.

Note: The wellbore radius, permeability, density and viscosity are fixed.


"""
from __future__ import annotations

import datetime
import logging
import math
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import runspecs
import tensorflow as tf
from ecl.eclfile.ecl_file import open_ecl_file
from mako.template import Template

import pyopmnearwell.utils.units as units
from pyopmnearwell.utils.formulas import peaceman_WI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


run_name: str = "ensemble_runs"
dirpath: str = os.path.dirname(os.path.realpath(__file__))
os.makedirs(os.path.join(dirpath, run_name), exist_ok=True)
ensemble_path = os.path.join(dirpath, run_name)


# Path to OPM with ml and co2brinepvt.
OPM_ML: str = "/home/peter/Documents/2023_CEMRACS/opm_ml"
CO2BRINEPVT: str = f"{OPM_ML}/build/opm-common/bin/co2brinepvt"

shutil.copyfile(
    os.path.join(dirpath, "TABLES.INC"),
    os.path.join(dirpath, run_name, "TABLES.INC"),
)
shutil.copyfile(
    os.path.join(dirpath, "CAKE.INC"),
    os.path.join(dirpath, run_name, "CAKE.INC"),
)

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

# Create the ensemble.
init_pressures_v, temperatures_v, permeabilities_v = np.meshgrid(
    runspecs.INIT_PRESSURES,
    runspecs.INIT_TEMPERATURES,
    runspecs.PERMEABILITIES,
    indexing="ij",
)
full_ensemble = np.stack(
    [
        init_pressures_v.flatten(),
        temperatures_v.flatten(),
        permeabilities_v.flatten(),
    ],
    axis=-1,
)

# Sample to reduce the number of ensemble members.
idx = np.random.randint(
    full_ensemble.shape[0],
    size=runspecs.NPOINTS,
)
ensemble = full_ensemble[idx, :]

# Create a deck file for each ensemble member.
mytemplate = Template(filename=os.path.join(dirpath, "RESERVOIR.mako"))
for i, member in enumerate(ensemble):
    pressure, temperature, permeability = member[0], member[1], member[2]
    var = {
        "init_pressure": pressure,
        "temperature": temperature,
        "permeability_x": permeability,
        "injection_rate": runspecs.INJECTION_RATE,
    }
    filledtemplate = mytemplate.render(**var)
    with open(
        os.path.join(ensemble_path, f"RESERVOIR{i}.DATA"),
        "w",
        encoding="utf8",
    ) as file:
        file.write(filledtemplate)  # type: ignore

# Store final grid cell pressures for each ensemble member.
pressures_lst: list[np.ndarray] = []

# Run OPM flow for each ensemble member.
for i in range(round(runspecs.NPOINTS / runspecs.NPRUNS)):
    os.system(
        f"{FLOW}"
        + f" {os.path.join(ensemble_path, f'RESERVOIR{runspecs.NPRUNS*i}.DATA')}"
        + f" --output-dir={os.path.join(ensemble_path, f'results{runspecs.NPRUNS*i}')} {FLAGS} & "
        f"{FLOW}"
        + f" {os.path.join(ensemble_path, f'RESERVOIR{runspecs.NPRUNS*i+1}.DATA')}"
        + f" --output-dir={os.path.join(ensemble_path, f'results{runspecs.NPRUNS*i+1}')} {FLAGS} & "
        f"{FLOW}"
        + f" {os.path.join(ensemble_path, f'RESERVOIR{runspecs.NPRUNS*i+2}.DATA')}"
        + f" --output-dir={os.path.join(ensemble_path, f'results{runspecs.NPRUNS*i+2}')} {FLAGS} & "
        f"{FLOW}"
        + f" {os.path.join(ensemble_path, f'RESERVOIR{runspecs.NPRUNS*i+3}.DATA')}"
        + f" --output-dir={os.path.join(ensemble_path, f'results{runspecs.NPRUNS*i+3}')} {FLAGS} & "
        f"{FLOW}"
        + f" {os.path.join(ensemble_path, f'RESERVOIR{runspecs.NPRUNS*i+4}.DATA')}"
        + f" --output-dir={os.path.join(ensemble_path, f'results{runspecs.NPRUNS*i+4}')} {FLAGS} & wait"
    )
    for j in range(runspecs.NPRUNS):
        with open_ecl_file(
            os.path.join(
                ensemble_path,
                f"results{runspecs.NPRUNS*i+j}",
                f"RESERVOIR{runspecs.NPRUNS*i+j}.UNRST",
            )
        ) as ecl_file:
            # Each pressure array has shape ``[number_report_steps, number_cells]``
            # Truncate the report step at starting time, as it just gives back
            # initial conditions. Take only every 3rd time step and every 4th radius to
            # reduce the dataset size.
            pressures_lst.append(np.array(ecl_file.iget_kw("PRESSURE"))[1::3, ::4])
            # We assume constant report step delta. The steps cannot be taken directly
            # from the ecl file, as the hours and minutes are missing.
            # Truncate the starting time.
            if i == 0 and j == 0:
                NUM_REPORT_STEPS: int = ecl_file.num_report_steps() - 1
                report_times: np.ndarray = np.linspace(
                    0,
                    (
                        ecl_file.report_dates[-1] - datetime.datetime(2000, 1, 1, 0, 0)
                    ).total_seconds()
                    / 3600,
                    NUM_REPORT_STEPS + 1,
                )[1::3]
        # Remove the run files and result folder (except for the first one).
        if runspecs.NPRUNS * i + j > 0:
            shutil.rmtree(
                os.path.join(ensemble_path, f"results{runspecs.NPRUNS*i+j}"),
            )
            os.remove(
                os.path.join(ensemble_path, f"RESERVOIR{runspecs.NPRUNS*i+j}.DATA")
            )
os.remove(os.path.join(ensemble_path, f"TABLES.INC"))
os.remove(os.path.join(ensemble_path, f"CAKE.INC"))


# Create feature tensor.
pressures_t = np.array(pressures_lst) * units.BAR_TO_PASCAL  # unit: [Pa];
# ``shape=(runspecs.NPOINTS, NUM_REPORT_STEPS // 3 + 1, 400)``
assert pressures_t.shape == (
    runspecs.NPOINTS,
    NUM_REPORT_STEPS // 3 + 1,
    100,
)

temperatures_t = np.array([member[1] for member in ensemble])  # unit: [°C]
assert temperatures_t.shape == (runspecs.NPOINTS,)
permeabilities_t = np.array([member[2] for member in ensemble])  # unit: [°C]
assert permeabilities_t.shape == (runspecs.NPOINTS,)

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

# Take only every 4th radius.
radii_t = radii_t[::4]
# Truncate the well cells and cells in the close proximity, as well as the outermost
# cell (pore-volume) to sanitize the dataset (WI close to cell behaves weirdly).
radii_t = radii_t[1:-1]
NUM_RADII: int = 98
assert radii_t.shape == (NUM_RADII,)


report_times_v, radii_v = np.meshgrid(report_times, radii_t, indexing="ij")
# NOTE: For ``temperatures_t`` and ``permeabilities_t``, we use ``np.repeat`` and for
# ``report_times_v`` and ``radii_v`` we use ``np.tile``.
assert (
    np.repeat(temperatures_t, (NUM_REPORT_STEPS // 3 + 1) * NUM_RADII).shape
    == pressures_t[..., 1:-1].flatten().shape
)
assert (
    np.tile(
        report_times_v.flatten(),
        runspecs.NPOINTS,
    ).shape
    == pressures_t[..., 1:-1].flatten().shape
)


features = np.stack(
    [
        pressures_t[..., 1:-1].flatten(),
        np.repeat(temperatures_t, (NUM_REPORT_STEPS // 3 + 1) * NUM_RADII),
        np.repeat(permeabilities_t, (NUM_REPORT_STEPS // 3 + 1) * NUM_RADII),
        np.tile(
            report_times_v.flatten(),
            runspecs.NPOINTS,
        ),
        np.tile(
            radii_v.flatten(),
            runspecs.NPOINTS,
        ),
    ],
    axis=-1,
)  # ``shape=(runspecs.NPOINTS * (NUM_REPORT_STEPS // 3 + 1) * NUM_RADII, 3)``

# Calculate the well indices from injection rate, cell pressures and bhp.
# Get the pressure value of the innermost cell, which equals the bottom hole pressure.
# Ignore the first report step, as it has constant pressure in the interval
bhp_t: np.ndarray = pressures_t[..., 0]

# Truncate the cell pressures the same way the radii were truncated.
# Multiply by 6 to account for the 6 times higher injection rate of a 360° well.
WI_data: np.ndarray = (
    runspecs.INJECTION_RATE
    * units.Q_per_day_to_Q_per_seconds
    / (bhp_t[..., None] - pressures_t[..., 1:-1])
) * 6  # unit: m^4*s/kg;
# ``shape=(runspecs.NPOINTS, NUM_REPORT_STEPS // 3 + 1, NUM_RADII, 1)``

# Store the features: pressures, radii, and targets: WI as a dataset.
ds = tf.data.Dataset.from_tensor_slices((features, WI_data.reshape((-1, 1))))
ds.save(os.path.join(ensemble_path, "pressure_radius_WI"))

# Plot pressure, distance - WI relationship vs Peaceman model
features = np.reshape(
    features,
    (runspecs.NPOINTS, -1, NUM_RADII, 5),
)

WI_analytical = (
    np.vectorize(peaceman_WI)(
        features[0, 0, 0, 2] * units.MILIDARCY_TO_M2 * units.MILIDARCY_TO_M2,
        radii_t,
        runspecs.WELL_RADIUS,
        runspecs.DENSITY,
        runspecs.VISCOSITY,
    )
    / runspecs.SURFACE_DENSITY
)

# Take data at the last report step.
for i, (feature, target) in enumerate(
    list(zip(features[:5, -1, ...], WI_data[:5, -1, ...]))
):
    plt.figure()
    plt.scatter(
        radii_t[::5],
        target[::5],
        label="data",
    )
    plt.plot(
        radii_t,
        WI_analytical,
        label="Peaceman",
    )
    plt.legend()
    plt.title(
        rf"$p={feature[-1][0]:.3e}\,[Pa]$ at $r={feature[-1][4]:.2f}\,[m]$"
        + "\n"
        + rf"$T={feature[0][1]:2f}\,[°C]$, $k={feature[0][2]:2f}\,[mD]$,"
        + rf"$t={feature[0][3]:2f}\,[h]$"
    )
    plt.xlabel(r"$r\,[m]$")
    plt.ylabel(r"$WI\,[m^4\cdot s/kg]$")
    plt.subplots_adjust(left=0.1)
    plt.savefig(os.path.join(dirpath, run_name, f"data_vs_Peaceman_{i}.png"))
    plt.show()
