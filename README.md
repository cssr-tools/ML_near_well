# ML_near_well
[![DOI]()]()
[![arXiv](https://img.shields.io/badge/arXiv-2601.11193-b31b1b.svg)](https://arxiv.org/abs/2601.11193)

Companion code for[*A machine-learned near-well model in OPM Flow, von Schultzendorff et al. (2024)*](https://www.earthdoc.org/content/papers/10.3997/2214-4609.202437033).

**ML_near_well** is a collection of runfiles for a machine-learned near-well model. The
key idea is to replace the analytical expression for well transmissibility from
Peaceman-type well models with a neural network. The network is trained on data from
fine-scale ensemble simulations of the near-well region under differing flow regimes.
This novel approach allows for flexible and accurate modeling of transient and
multiphase effects.

The ensemble simulations as well as tests of the final model are run in the open-source
reservoir simulator OPM Flow. In addition, our code uses the
[pyopmnearwell](https://github.com/cssr-tools/pyopmnearwell) package to run near-well
ensemble simulations, extract data sets, and train models. The implementation of the ML
near-well model in OPM Flow closely follows the approach for the
[Hybrid Newton method](https://github.com/cssr-tools/hyopml/tree/main). The files
``MLNearWellConfig.hpp`` and ``MLNearWellConfig.cpp`` are modified versions of the
corresponding hybrid Newton equivalents.

**Note:** All scripts were run with ``OPM Flow 2025.10``,
``python 3.10.12``,  and the python packages specified in ``requirements_full.txt``.

# Installation
To install and run with Docker:
```bash
git clone https://github.com/cssr-tools/ML_near_well -b reproducible
cd ML_near_well
docker-compose up
```

or, if you prefer to install on your own machine,

1. Clone this repo
   ``git clone --branch reproducible https://github.com/cssr-tools/ML_near_well/``
2. Create a virtual environment (e.g., with ``conda``), navigate to the local repo, and
   install the dependencies with
   ``pip install -r requirements.txt``
   or (if you run into errors at any point)
   ``pip install -r requirements_full.txt``
3. Clone ``pyopmnearwell``
   ``git clone --branch 2024-08_ML_near_well_article https://github.com/cssr-tools/pyopmnearwell/``,
   navigate to the local repo, and install ``pyopmnearwell`` with
   ``pip install .``
4. Install OPM Flow or build from source https://opm-project.org/?page_id=36 (needed to
   run the ensemble scripts).
5. The paper runscripts read ``OPM_PATH`` and ``FLOW_PATH`` environment variables.
6. By default, local runs fall back to ``OPM=/usr`` and ``FLOW=/usr/bin/flow``. In the
   Docker image, ``OPM_PATH`` points to the source tree under ``/opt/opm_src``.
7. If needed, update path logic in:
```
h2o/runspecs.py
co2_2d/runspecs.py
co2_3d/runspecs.py
```

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
To reproduce the paper results and figures locally (requires that ``python3.10`` is an
available command), run:
```
bash runscripts/run.bash
```

With Docker, the image default command already runs the same script:
```
docker run --rm ml_near_well:0.1
```
or
```
docker compose up --build
```
Alternatively, you can run each of the examples individually, e.g.,:
```
cd examples/h2o_extended
python3.10 main.py
```
Results 1, 2, and 3 in the paper correspond to ``h2o``, ``co2_2d``,
and ``co2_3d``.

# Citing
If you use either all or part of of the code in this repository, we kindly ask you to
cite the following reference:

P. von Schultzendorff, T. H. Sandve, B. Kane, D. Landa-Marbán, J. W. Both, and J. M. Nordbotten, “A Machine-Learned Near-Well Model in OPM Flow”, presented at ECMOR 2024, European Association of Geoscientists & Engineers, Sep. 2024, pp. 1–23. doi: 10.3997/2214-4609.202437033.
