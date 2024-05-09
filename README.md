# ml_near_well
**ml_near_well** is a collection of runfiles for a machine learned near well model. The
key idea is to replace the analytical expression for well transmissibility from
Peaceman-type well models with a neural network. The network is trained on data from
fine-scale ensemble simulations of the near-well region under differing flow regimes.
This novel approach allows for flexible and accurate modeling of transient and
multiphase effects.

The accompanying paper is [*A machine-learned near-well model in OPM Flow*]().

The ensemble simulations as well as tests of the final model are run in the open-source
reservoir simulator OPM Flow. In additiona, our code uses the
[pyopmnearwell](https://github.com/cssr-tools/pyopmnearwell) package for near-well
ensemble simulations and model training and the [OPM Flow - neural network
framework](https://github.com/fractalmanifold/ml_integration2opm) for integration of
neural networks into OPM Flow. 

![Figure]

# Installation
TODO: Fix installation steps.
1. Create a virtual environment (e.g., with ``conda``) and install the dependencies with
   ``pip install -r requirements.txt``.
2. Clone https://github.com/cssr-tools/pyopmnearwell/tree/development, go to the local
   repo, and install with ``pip install .``-.
3. Install OPM or build from source https://opm-project.org/?page_id=36 (needed to run
   the ensemble scripts).
4. Build OPM with ML integration from source
   https://github.com/fractalmanifold/ml_integration2opm/tree/main (needed to run the
   integration scripts).
5. Update the paths to OPM and OPM with ML integration in the runscripts.
```
h2o_extended/runspecs.py
co2_2d_extended/runspecs.py
co2_3d_extended/runspecs.py
```
6. Clone this repo ``git clone ...``.

# Usage


# Reproduce results
To reproduce the paper results and figures, run these commands:
```
cd examples
bash run.bash
```
Alternatively, you can run each of the examples individually, e.g.,:
```
cd examples/h2o_extended
python main.py
```

# Citing
If you use  (either all or a part of) any of the code in this repository, we kindly ask
you to cite the following reference:

TODO: Fix reference.
von Schultzendorff, P., Sandve, T.H., Kane, B., Landa-Marban, D., Both,
J.W., Nordbotten, J. "A machine-learned near-well model in OPM Flow" (2024), To be
published