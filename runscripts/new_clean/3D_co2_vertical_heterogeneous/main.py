import pathlib

import numpy as np
import ..utils

from pyopmnearwell.ml import ensemble

import runspecs

def main():
    ensemble_dirname: pathlib.Path = dirname / runspecs_ensemble["name"]
    data_dirname: pathlib.Path = dirname / f"dataset_{runspecs_ensemble['name']}"

    ensemble_dirname.mkdir(exist_ok=True)
    data_dirname.mkdir(exist_ok=True)

    extracted_data: np.ndarray = utils.full_ensemble(runspecs_ensemble, ensemble_dirname)
    np.save(str(ensemble_dirname / "features"), extracted_data)

    extracted_data = np.load(str(ensemble_dirname / "features.npy"))
    dataset = Dataset(extracted_data, runspecs_ensemble, 5)
    features, targets = dataset.create_ds(
        ensemble_dirname, step_size_x=1, step_size_t=3
    )
    ensemble.store_dataset(features, targets, data_dirname)


if __name__ == "__main__":
    main()
