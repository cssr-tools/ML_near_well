import pathlib

from pyopmnearwell.ml import integration
from runspecs import runspecs_integration
from runspecs import trainspecs_2 as trainspecs

dirname: pathlib.Path = pathlib.Path(__file__).parent

nn_dirname: pathlib.Path = dirname / f"nn_ensemble_1_{trainspecs['name']}"
integration_dirname: pathlib.Path = (
    dirname / f"integration_ensemble_1_{trainspecs['name']}"
)
integration_dirname.mkdir(parents=True, exist_ok=True)

runspecs_integration["variables"].update(
    {"ML_MODEL_PATH": [nn_dirname / "WI.model", "", ""]}
)

integration.recompile_flow(
    nn_dirname / "scalings.csv",
    runspecs_integration["constants"]["OPM_ML"],
    StandardWell_impl_template="co2_local_stencil",
    StandardWell_template="local_stencil",
    stencil_size=3,
    local_feature_names=["pressure", "saturation", "permeability"],
)
integration.run_integration(
    runspecs_integration, integration_dirname, dirname / "integration.mako"
)
