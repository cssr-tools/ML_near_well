from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from nn import restructure_data
from pyopmnearwell.ml import ensemble, utils
from pyopmnearwell.utils import units
from runspecs import runspecs_ensemble, trainspecs
from upscale_ex import CO2_3D_upscaler_extrapolation

dirname: pathlib.Path = pathlib.Path(__file__).parent

sys.path.append(str(dirname / ".."))
from utils import read_existing_ensemble_results

SEED: int = 19123
utils.enable_determinism(SEED)

ensemble_dir: pathlib.Path = dirname / "ensemble_test"
data_dir: pathlib.Path = dirname / "dataset_ex"
data_stencil_dir: pathlib.Path = dirname / "dataset_stencil_ex"
nn_dir: pathlib.Path = dirname / "nn"

ensemble_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
data_stencil_dir.mkdir(parents=True, exist_ok=True)
nn_dir.mkdir(parents=True, exist_ok=True)

ANGLE: float = math.pi / 3

# # Run ensemble and extract data.
if True:
    print("=== BEFORE full_ensemble ===", flush=True)
    extracted_data = read_existing_ensemble_results(
        ensemble_dir,
        ecl_keywords=["PRESSURE", "SGAS", "FLOGASI+"],
        summary_keywords=["FGIT", "WGIR:INJ0"],
        keyword_scalings={"PRESSURE": units.BAR_TO_PASCAL},
    )
    print("=== AFTER full_ensemble ===", flush=True)
    print(
        f"extracted_data shape={extracted_data.shape} dtype={extracted_data.dtype}",
        flush=True,
    )
    np.save(str(ensemble_dir / "features"), extracted_data)
    print("=== AFTER np.save(features) ===", flush=True)
 
if True:
    extracted_data = np.load(str(ensemble_dir / "features.npy"), mmap_mode="r")
    upscaler: CO2_3D_upscaler_extrapolation = CO2_3D_upscaler_extrapolation(
        extracted_data, runspecs_ensemble, data_dim=5,angle=ANGLE
    )
    print("\n=== STAGE: upscaler.create_ds START ===", flush=True)
    features, targets = upscaler.create_ds(ensemble_dir, step_size_x=12, step_size_t=1, keep_xcells=142)
    # Fjern de to innerste punktene nær brønnen
    features = features[..., 2:, :]
    targets = targets[..., 2:]
    
    
    
    print(f"create_ds shapes: features={features.shape}, targets={targets.shape}", flush=True)
    print(f"create_ds dtypes: features={features.dtype}, targets={targets.dtype}", flush=True)
    print("=== STAGE: upscaler.create_ds DONE ===", flush=True)
    print("\n=== STAGE: store_dataset(raw) START ===", flush=True)
    ensemble.store_dataset(features, targets, data_dir)
    print("=== STAGE: store_dataset(raw) DONE ===", flush=True)
    print("\n=== STAGE: restructure_data START ===", flush=True)
    

    restructure_data(data_dir, data_stencil_dir, trainspecs, stencil_size=3)
    
    print("\n=== EXTRAPOLATION DATASET READY ===", flush=True)
    print(f"Read existing results from: {ensemble_dir}", flush=True)
    print(f"Saved raw dataset to: {data_dir}", flush=True)
    print(f"Saved stencil dataset to: {data_stencil_dir}", flush=True)
    print("No training was run.", flush=True)