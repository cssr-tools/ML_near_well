# SPDX-FileCopyrightText: 2023 NORCE
# SPDX-FileCopyrightText: 2023 UiB
# SPDX-License-Identifier: GPL-3.0
""""Run a CO2 injection on reservoir-scale in flow with the Peaceman well model, machine
learned well model and a finescale simulation."""

from __future__ import annotations

import csv
import os
import shutil
from typing import Any

import numpy as np
import runspecs
from mako.template import Template

import pyopmnearwell.utils.units as units

# Path to the simulation *.makos.
PATH: str = os.path.dirname(os.path.realpath(__file__))

# Path to OPM with ml and flow.
OPM_ML: str = "/home/peter/Documents/2023_CEMRACS/opm_ml"
FLOW: str = f"{OPM_ML}/build/opm-simulators/bin/flow_gaswater_dissolution_diffuse"

# Get the scaling and write it to the C++ mako that integrates nn into OPM.
feature_min: list[float] = []
feature_max: list[float] = []
with open(os.path.join(PATH, "model_pressure_radius_WI", "scales.csv")) as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=["variable", "min", "max"])
    for row in reader:
        match row["variable"]:
            case "pressure":
                feature_min.append(float(row["min"]))
                feature_max.append(float(row["max"]))
            case "radius":
                feature_min.append(float(row["min"]))
                feature_max.append(float(row["max"]))
            case "WI":
                target_min: float = float(row["min"])
                target_max: float = float(row["max"])

var: dict[str, Any] = {
    "xmin": feature_min,
    "xmax": feature_max,
    "ymin": target_min,
    "ymax": target_max,
}
mytemplate = Template(filename=os.path.join(PATH, "StandardWell_impl.mako"))
filledtemplate = mytemplate.render(**var)
with open(
    f"{OPM_ML}/opm-simulators/opm/simulators/wells/StandardWell_impl.hpp",
    "w",
    encoding="utf8",
) as file:
    file.write(filledtemplate)

shutil.copyfile(
    os.path.join(PATH, "StandardWell.hpp"),
    f"{OPM_ML}/opm-simulators/opm/simulators/wells/StandardWell.hpp",
)
# Recompile flow.
os.chdir(f"{OPM_ML}/build/opm-simulators")
os.system("make -j5 flow_gaswater_dissolution_diffuse")


INIT_PRESSURE: float = (
    (runspecs.INIT_PRESSURE_MIN + runspecs.INIT_PRESSURE_MAX) / 2 * units.BAR_TO_PASCAL
)  # unit: [Pa]; average of training pressure.
# NOTE: During training we train on the wellblock pressure, not on the pressure of the
# entire reservoir.

# Write the configuration files for the comparison in the 3D reservoir
var = {
    "flow": FLOW,
    "perm": runspecs.PERMEABILITY,
    "pressure": INIT_PRESSURE,
    "temperature": runspecs.TEMPERATURE,
    "radius": runspecs.WELL_RADIUS,
    "reservoir_size": runspecs.RESERVOIR_SIZE,
    "rate": runspecs.INJECTION_RATE
    * runspecs.SURFACE_DENSITY
    * 6,  # Transform from m^3/day to kg^/day. Multiply by 6 to account for the 360°
    # well in comparison to the 60° well in the ensemble.
    "pwd": PATH,
}

# Use our pyopmnearwell friend to run the 3D simulations and compare the results
os.chdir(os.path.join(PATH, "tests"))
for name in [
    "3d_flow_wellmodel",
    "3d_ml_wellmodel",
    "3d_finescale_wellmodel",
]:
    mytemplate = Template(filename=os.path.join(f"h2o_{name}.mako"))
    filledtemplate = mytemplate.render(**var)
    with open(
        os.path.join(f"h2o_{name}.txt"),
        "w",
        encoding="utf8",
    ) as file:
        file.write(filledtemplate)
    os.system(f"pyopmnearwell -i h2o_{name}.txt -o h2o_{name}")
