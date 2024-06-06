# ML_near_well
**ML_near_well** is a collection of runfiles for a machine-learned near-well model. The
key idea is to replace the analytical expression for well transmissibility from
Peaceman-type well models with a neural network. The network is trained on data from
fine-scale ensemble simulations of the near-well region under differing flow regimes.
This novel approach allows for flexible and accurate modeling of transient and
multiphase effects.

The accompanying paper is [*A machine-learned near-well model in OPM Flow*](), to be
published later in 2024.

The ensemble simulations as well as tests of the final model are run in the open-source
reservoir simulator OPM Flow. In addition, our code uses the
[pyopmnearwell](https://github.com/cssr-tools/pyopmnearwell) package for near-well
ensemble simulations and model training and the [OPM Flow - neural network framework]()
for integration of neural networks into OPM Flow. 

**Note:** The latter is not publicly available yet (as of 06.06.2024), without it the
code in this repository will only run partly. As soon as everything is available, this note will be removed.

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
You can reproduce the paper results as described below. To create and integrate your own
near-well model, follow the structure of the examples. The workflow consists of four
steps:
1. Run an ensemble of radial fine-scale near-well simulations.
   ``pyopmnearwell.ml.ensemble`` provides useful functions.
2. Extract and upscale features and targets to create a dataset.
   ``pyopmnearwell.ml.ensemble`` and ``pyopmnearwell.ml.upscale`` provide useful
   functions.
3. Train a neural network in Tensorflow. ``pyopmnearwell.ml.nn`` and
   ``ML_near_well.utils`` provide useful functions.
4. Integrate the network into OPM Flow and run a full simulation.
   ``pyopmnearwell.ml.integration`` provides useful functions.

NOTE: At the moment, some hardcoded hacks are needed to make everything work. Make sure
that you use the right values to get correct results.
- The total injected volume inside OPM Flow is calculated by multiplying elapsed time
  with injection rate. The injection rate is hardcoded for each model and needs to be
  adjusted in ``standardwell_impl.mako`` inside the ``wellIndexEval`` function. (This is
  relevant for the CO2 examples.)
- The scaling of outputs and inputs for the NN is done inside OPM Flow. However, the
  scaling values are hardcoded and OPM Flow needs to be recompiled each time the model
  changes. ``pyopmnearwell`` provides some helper functions (in ``ml.nn`` and
  ``ml.integration``) that automatically store these values, fill them into the
  ``standardwell_impl.mako`` templates and recompile Flow.
  In the release version of OPM Flow - NN version, scaling values will be stored
  directly inside the neural network, such that this procedure is no longer needed.
- OPM Flow does not support radial simulations. Instead the near-well ensemble
  simulations are run on a triangle shaped domain. The results correspond then to radii
  adjusted with ``pyopmnearwell.utils.formulas.pyopmnearwell_correction`` on a radial
  grid of the same angle as the triangle. Afterwards, some results/values, such as
  injection rate, still need to be adjusted to a full 360° well.

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
Results 1, 2, and 3 in the paper correspond to ``h2o``, ``co2_2d``,
and ``co2_3d``.

# Citing
If you use either all or part of of the code in this repository, we kindly ask you to
cite the following reference:

TODO: Fix reference.
von Schultzendorff, P., Sandve, T.H., Kane, B., Landa-Marban, D., Both,
J.W., Nordbotten, J. "A machine-learned near-well model in OPM Flow" (2024), To be
published