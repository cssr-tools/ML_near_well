# SPDX-FileCopyrightText: 2023 NORCE
# SPDX-FileCopyrightText: 2023 UiB
# SPDX-License-Identifier: GPL-3.0
""""Run a CO2 injection in flow both with the Peaceman well model and machine learned
well model."""

from __future__ import annotations

import csv
import os
import shutil
from typing import Any

import runspecs
from mako.template import Template

import pyopmnearwell.utils.units as units

# Path to the simulation *.makos.
PATH: str = os.path.dirname(os.path.realpath(__file__))

# Path to OPM with ml and flow.
OPM_ML: str = "/home/peter/Documents/2023_CEMRACS/opm_ml"
FLOW: str = f"{OPM_ML}/build/opm-simulators/bin/flow_gaswater_dissolution_diffuse"

# Test case 1:
INIT_PRESSURE: float = (
    runspecs.INIT_PRESSURE_MIN + 5
) * units.BAR_TO_PASCAL  # unit: [Pa];
PERM_1: float = 10 * runspecs.PERMEABILITY_MIN * units.MILIDARCY_TO_M2  # unit: [m^2];
PERM_2: float = 12 * runspecs.PERMEABILITY_MIN * units.MILIDARCY_TO_M2  # unit: [m^2];
PERM_3: float = 14 * runspecs.PERMEABILITY_MIN * units.MILIDARCY_TO_M2  # unit: [m^2];
INIT_TEMPERATURE: float = 0.5 * (
    runspecs.INIT_TEMPERATURE_MIN + runspecs.INIT_TEMPERATURE_MAX
)  # unit: [°C];

# Get the scaling and write it to the C++ mako that integrates nn into OPM.
feature_min: list[float] = []
feature_max: list[float] = []
with open(os.path.join(PATH, "ml_model", "scales.csv")) as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=["variable", "min", "max"])
    for row in reader:
        match row["variable"]:
            case "pressure":
                feature_min.append(float(row["min"]))
                feature_max.append(float(row["max"]))
            case "temperature":
                feature_min.append(float(row["min"]))
                feature_max.append(float(row["max"]))
            case "permeability":
                feature_min.append(float(row["min"]))
                feature_max.append(float(row["max"]))
            case "time":
                feature_min.append(float(row["min"]))
                feature_max.append(float(row["max"]))
            case "radius":
                feature_min.append(float(row["min"]))
                feature_max.append(float(row["max"]))
            case "WI":
                target_min: float = float(row["min"])
                target_max: float = float(row["max"])

var_1: dict[str, Any] = {
    "xmin": feature_min,
    "xmax": feature_max,
    "ymin": target_min,
    "ymax": target_max,
    "temperature": INIT_TEMPERATURE,
}
mytemplate = Template(filename=os.path.join(PATH, "StandardWell_impl.mako"))
filledtemplate = mytemplate.render(**var_1)
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
# os.system("make -j5 flow_gaswater_dissolution_diffuse")

# Write the configuration files for the comparison in the 3D reservoir
var_2: dict[str, Any] = {
    "flow": FLOW,
    "perm1": PERM_1,
    "perm2": PERM_2,
    "perm3": PERM_3,
    "pressure": INIT_PRESSURE,
    "temperature": INIT_TEMPERATURE,
    "radius": runspecs.WELL_RADIUS,
    "reservoir_size": runspecs.RESERVOIR_SIZE,
    "rate": runspecs.INJECTION_RATE,
    "pwd": PATH,
}

# Use our pyopmnearwell friend to run the 3D simulations and compare the results
os.chdir(os.path.join(PATH, "test_1"))
for name in [
    # "3d_scale_coarse",
    # "3d_scale_coarse_ml",
    # "3d_scale_middle",
    # "3d_scale_fine",
    # "3d_scale_super_fine",
]:
    mytemplate = Template(filename=f"co2_{name}.mako")
    filledtemplate = mytemplate.render(**var_2)
    with open(
        os.path.join(f"co2_{name}.txt"),
        "w",
        encoding="utf8",
    ) as file:
        file.write(filledtemplate)
    os.system(f"pyopmnearwell -i co2_{name}.txt -o co2_{name}")

# Test case 3:
var_2.update({"reservoir_size": 525})

os.chdir(os.path.join(PATH, "test_3"))
for name in [
    # "3d_scale_coarse",
    # "3d_scale_coarse_ml",
    # "3d_scale_middle",
    # "3d_scale_fine",
]:
    mytemplate = Template(filename=f"co2_{name}.mako")
    filledtemplate = mytemplate.render(**var_2)
    with open(
        os.path.join(f"co2_{name}.txt"),
        "w",
        encoding="utf8",
    ) as file:
        file.write(filledtemplate)
    os.system(f"pyopmnearwell -i co2_{name}.txt -o co2_{name}")


# Test case 2:
INIT_PRESSURE: float = (
    runspecs.INIT_PRESSURE_MIN + 15
) * units.BAR_TO_PASCAL  # unit: [Pa];
INIT_TEMPERATURE: float = 0.6 * (
    runspecs.INIT_TEMPERATURE_MIN + runspecs.INIT_TEMPERATURE_MAX
)  # unit: [°C];

var_1.update({"temperature": INIT_TEMPERATURE})
mytemplate = Template(filename=os.path.join(PATH, "StandardWell_impl.mako"))
filledtemplate = mytemplate.render(**var_1)
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
# os.system("make -j5 flow_gaswater_dissolution_diffuse")

var_2.update(
    {
        "reservoir_size": runspecs.RESERVOIR_SIZE,
        "pressure": INIT_PRESSURE,
        "temperature": INIT_TEMPERATURE,
    }
)

# Use our pyopmnearwell friend to run the 3D simulations and compare the results
os.chdir(os.path.join(PATH, "test_2"))
for name in [
    # "3d_scale_coarse",
    # "3d_scale_coarse_ml",
    # "3d_scale_middle",
    # "3d_scale_fine",
]:
    mytemplate = Template(filename=f"co2_{name}.mako")
    filledtemplate = mytemplate.render(**var_2)
    with open(
        os.path.join(f"co2_{name}.txt"),
        "w",
        encoding="utf8",
    ) as file:
        file.write(filledtemplate)
    os.system(f"pyopmnearwell -i co2_{name}.txt -o co2_{name}")


# Test case 4 -> Same as 3 but adjust for radius taken at left/right cell boundaries
# during training.
mytemplate = Template(filename=os.path.join(PATH, "StandardWell_impl_adjusted_r.mako"))
filledtemplate = mytemplate.render(**var_1)
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
    runspecs.INIT_PRESSURE_MIN + 5
) * units.BAR_TO_PASCAL  # unit: [Pa];
PERM_1: float = 10 * runspecs.PERMEABILITY_MIN * units.MILIDARCY_TO_M2  # unit: [m^2];
PERM_2: float = 12 * runspecs.PERMEABILITY_MIN * units.MILIDARCY_TO_M2  # unit: [m^2];
PERM_3: float = 14 * runspecs.PERMEABILITY_MIN * units.MILIDARCY_TO_M2  # unit: [m^2];
INIT_TEMPERATURE: float = 0.5 * (
    runspecs.INIT_TEMPERATURE_MIN + runspecs.INIT_TEMPERATURE_MAX
)  # unit: [°C];

var_2: dict[str, Any] = {
    "flow": FLOW,
    "perm1": PERM_1,
    "perm2": PERM_2,
    "perm3": PERM_3,
    "pressure": INIT_PRESSURE,
    "temperature": INIT_TEMPERATURE,
    "radius": runspecs.WELL_RADIUS,
    "reservoir_size": 525,
    "rate": runspecs.INJECTION_RATE,
    "pwd": PATH,
}

os.chdir(os.path.join(PATH, "test_4"))
for name in [
    # "3d_scale_coarse",
    "3d_scale_coarse_ml",
    # "3d_scale_middle",
    # "3d_scale_fine",
]:
    mytemplate = Template(filename=f"co2_{name}.mako")
    filledtemplate = mytemplate.render(**var_2)
    with open(
        os.path.join(f"co2_{name}.txt"),
        "w",
        encoding="utf8",
    ) as file:
        file.write(filledtemplate)
    os.system(f"pyopmnearwell -i co2_{name}.txt -o co2_{name}")
