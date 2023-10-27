""""Run nearwell H2O injection simulations in OPM-Flow for an ensemble of varying
initial pressures and construct a dataset containing cell pressures and
equivalent well-radii as features and well indices as targets.

Features:
    1. pressure [Pa]: Measured each report step at the equivalent well radius.
    2. radius [m]: Gives the equivalent well radius.

Targets:
    1. WI [[m^4*s/kg]]: Calculated at the given radius.

Note: The wellbore radius, permeability, density and viscosity are fixed.


"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess

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

# Create a deck file for each ensemble member
mytemplate = Template(filename=os.path.join(dirpath, "RESERVOIR.mako"))
for i, pressure in enumerate(runspecs.INIT_PRESSURES):
    var = {
        "permeability_x": runspecs.PERMEABILITY,
        "init_pressure": pressure,
        "temperature": runspecs.TEMPERATURE,
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
pressures: list[np.ndarray] = []
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
            # Take the pressure at the last report step.
            pressures.append(np.array(ecl_file.iget_kw("PRESSURE"))[-1])
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


# Assemble inputs for the feature tensor.
pressures_t = (
    np.array(pressures) * units.BAR_TO_PASCAL
)  # ``shape=(runspecs.NPOINTS, num_report_steps/5, 400)``, unit: [Pa]

# Calculate cell center radii.
with open(os.path.join(dirpath, "CAKE.INC"), "r") as radii_file:
    lines: list[str] = radii_file.readlines()[9:410]
    assert len(lines) == 401
    radii: np.ndarray = np.array(
        list(
            map(
                lambda x: float(x.strip("\n").split()[0]),
                lines,
            )
        )
    )
    inner_radii: np.ndarray = radii[:-1]
    outer_radii: np.ndarray = radii[1:]
    radii_t: np.ndarray = (inner_radii + outer_radii) / 2  # unit: [m]
inner_radii: np.ndarray = radii[:-1]
outer_radii: np.ndarray = radii[1:]
radii_t: np.ndarray = (inner_radii + outer_radii) / 2  # unit: [m]

# Truncate the well cells and cells in the close proximity, as well as the outermost
# cell (pore-volume) to sanitize the dataset (WI close to cell behaves weirdly).
radii_t = radii_t[4:-1]
assert radii_t.shape == (395,)

features = np.stack(
    [
        pressures_t[..., 4:-1].flatten(),
        np.tile(radii_t, runspecs.NPOINTS),
    ],
    axis=-1,
)  # ``shape=(runspecs.NPOINTS * num_report_steps * 395, 2)``

# Calculate the well indices from injection rate, cell pressures and bhp.
# Get the pressure value of the innermost cell, which equals the bottom hole pressure.
# Ignore the first report step, as it has constant pressure in the interval
bhp_t: np.ndarray = pressures_t[..., 0]

# Truncate the cell pressures the same way the radii were truncated.
# NOTE: Multiply by 6 to account for the 6 times higher injection rate of a 360° well.
WI_data: np.ndarray = (
    runspecs.INJECTION_RATE
    * units.Q_per_day_to_Q_per_seconds
    / (bhp_t[..., None] - pressures_t[..., 4:-1])
) * 6  # unit: m^4*s/kg; ``shape=(runspecs.NPOINTS, num_report_steps/5, 395, 1)``

# Store the features: pressures, radii, and targets: WI as a dataset.
ds = tf.data.Dataset.from_tensor_slices((features, WI_data.flatten()[..., None]))
ds.save(os.path.join(ensemble_path, "pressure_radius_WI"))

# Plot pressure, distance - WI relationship vs Peaceman model

# Calculate density and viscosity
OPM_ML: str = "/home/peter/Documents/2023_CEMRACS/opm_ml"
CO2BRINEPVT: str = os.path.join(OPM_ML, "build/opm-common/bin/co2brinepvt")

with subprocess.Popen(
    [
        CO2BRINEPVT,
        "density",
        "brine",
        str(pressure),
        str(runspecs.TEMPERATURE + units.CELSIUS_TO_KELVIN),
    ],
    stdout=subprocess.PIPE,
) as proc:
    density: float = float(proc.stdout.read())
with subprocess.Popen(
    [
        CO2BRINEPVT,
        "viscosity",
        "brine",
        str(pressure),
        str(runspecs.TEMPERATURE + units.CELSIUS_TO_KELVIN),
    ],
    stdout=subprocess.PIPE,
) as proc:
    viscosity: float = float(proc.stdout.read())

WI_analytical = (
    np.vectorize(peaceman_WI)(
        runspecs.PERMEABILITY * units.MILIDARCY_TO_M2,
        radii_t,
        runspecs.WELL_RADIUS,
        density,
        viscosity,
    )
    / runspecs.SURFACE_DENSITY
)

features = np.reshape(features, (runspecs.NPOINTS, 395, 2))
for feature, target in list(zip(features[:3, ...], WI_data[:3, ...])):
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
        rf"$p={feature[20][0]:.3e}\,[Pa]$ Peaceman at $r={feature[20][1]:.2f}\,[m]$"
    )
    plt.xlabel(r"$r\,[m]$")
    plt.ylabel(r"$WI\,[m^4\cdot s/kg]$")
    plt.savefig(
        os.path.join(dirpath, run_name, f"data_vs_Peaceman_p_{feature[20][0]:.3e}.png")
    )
    plt.show()