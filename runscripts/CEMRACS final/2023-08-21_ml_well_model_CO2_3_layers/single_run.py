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


run_name: str = "single_run"
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

# Create a deck file for each ensemble member.
mytemplate = Template(filename=os.path.join(dirpath, "RESERVOIR.mako"))
pressure, temperature, permeability = 55, 30, 1e-12 * units.M2_TO_MILIDARCY
var = {
    "init_pressure": pressure,
    "temperature": temperature,
    "permeability_x": permeability,
    "injection_rate": runspecs.INJECTION_RATE,
}
filledtemplate = mytemplate.render(**var)
with open(
    os.path.join(ensemble_path, f"RESERVOIR.DATA"),
    "w",
    encoding="utf8",
) as file:
    file.write(filledtemplate)  # type: ignore

# Run OPM flow .
os.system(
    f"{FLOW}"
    + f" {os.path.join(ensemble_path, f'RESERVOIR.DATA')}"
    + f" --output-dir={os.path.join(ensemble_path, f'results')} {FLAGS}"
)
