# ml_near_well_model
This is a collection of runfiles for an machine learned near well model based on
[pyopmnearwell](https://daavid00.github.io/pyopmnearwell/introduction.html) and [a fork of OPM](https://github.com/fractalmanifold/ml_integration2opm).

# Installation
1. Create a virtual environment and install all requirements.
2. Clone https://github.com/daavid00/pyopmnearwell/tree/development and install.
3. Install OPM or build from source https://opm-project.org/?page_id=36 (to run the
   ensemble scripts).
4. Build OPM with ML integration from source
   https://github.com/fractalmanifold/ml_integration2opm/tree/main (to run the
   integration scripts).
5. Update the paths to OPM and OPM with ML integration in the runscripts.