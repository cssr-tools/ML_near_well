# SPDX-FileCopyrightText: 2023 NORCE
# SPDX-FileCopyrightText: 2023 UiB
# SPDX-License-Identifier: GPL-3.0
""""Run a CO2 injection in flow both with the Peaceman well model and machine learned
well model."""

import csv
import math
import os
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ecl.eclfile import EclFile
from mako.template import Template

PATH: str = os.path.dirname(os.path.realpath(__file__))

# Give the full path to PYOPMNEARWELL and model parameters (ranges and inj rate based on
# the csp11b model)
OPM_ML: str = "/home/peter/Documents/2023_CEMRACS/opm_ml"
FLOW: str = f"{OPM_ML}/build/opm-simulators/bin/flow_gaswater_dissolution_diffuse"

# Get the scaling and write it to the c++ mako.
feature_min: list[float] = []
feature_max: list[float] = []
with open(os.path.join(PATH, "model_permeability_radius_WI", "scales.csv")) as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=["variable", "min", "max"])
    for row in reader:
        match row["variable"]:
            case "permeability":
                feature_min.append(float(row["min"]))
                feature_min.append(float(row["max"]))
            case "init_pressure":
                feature_min.append(float(row["min"]))
                feature_min.append(float(row["max"]))
            case "radius":
                feature_min.append(float(row["min"]))
                feature_min.append(float(row["max"]))
            case "WI":
                target_min: float = float(row["min"])
                target_max: float = float(row["max"])

var: dict[str, Any] = {
    "xmin": feature_min,
    "xmax": feature_max,
    "ymin": target_min,
    "ymax": target_max,
}
mytemplate = Template(filename="StandardWell_impl.mako")
filledtemplate = mytemplate.render(**var)
with open(
    f"{OPM_ML}/opm-simulators/opm/simulators/wells/StandardWell_impl.hpp",
    "w",
    encoding="utf8",
) as file:
    file.write(filledtemplate)
pwd = os.getcwd()
os.chdir(f"{OPM_ML}/build/opm-simulators")
os.system("make -j5 flow_gaswater_dissolution_diffuse")
os.chdir(f"{pwd}")


PERMEABILITY: float = 1e-12  # k between 1e-13 to 1e-12 m^2 during training.
INIT_PRESSURE: float = 9e6  # p between 7e6 and 1e7 Pascal during training.
WELLRADIUS: float = 0.25  # Fixed during training.
INJECTION_RATE: float = 5.352087e3  # Fixed during training.

# Write the configuration files for the comparison in the 3D reservoir
var = {
    "flow": FLOW,
    "perm": PERMEABILITY,
    "pressure": INIT_PRESSURE,
    "radius": WELLRADIUS,
    "rate": INJECTION_RATE,
    "pwd": os.getcwd(),
}

for name in ["3d_flow_wellmodel", "3d_ml_wellmodel"]:
    mytemplate = Template(filename=os.path.join(PATH, f"co2_{name}.mako"))
    filledtemplate = mytemplate.render(**var)
    with open(
        os.path.join(PATH, f"co2_{name}.txt"),
        "w",
        encoding="utf8",
    ) as file:
        file.write(filledtemplate)

# Use our pyopmnearwell friend to run the 3D simulations and compare the results
os.chdir(PATH)
# os.system("pyopmnearwell -i co2_3d_flow_wellmodel.txt -o co2_3d_flow_wellmodel")
os.system("pyopmnearwell -i co2_3d_ml_wellmodel.txt -o co2_3d_ml_wellmodel")
os.system("pyopmnearwell -c compare")
