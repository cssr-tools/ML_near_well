import pathlib

import seaborn as sns
from pyopmnearwell.ml import analysis
from runspecs import runspecs_ensemble_2 as runspecs_ensemble
from runspecs import trainspecs_2 as trainspecs
from tensorflow import keras

dirname: pathlib.Path = pathlib.Path(__file__).parent

FEATURE_TO_INDEX: dict[str, int] = {
    "pressure_upper": 0,
    "pressure": 1,
    "pressure_lower": 2,
    "saturation_upper": 3,
    "saturation": 4,
    "saturation_lower": 5,
    "permeability_upper": 6,
    "permeability": 7,
    "permeability_lower": 8,
    "radius": 9,
    "total_injected_volume": 10,
    "PI_analytical": 11,
}

# Set seaborn style.
sns.set_theme(context="paper", style="whitegrid")

nn_dirname: pathlib.Path = pathlib.Path(dirname) / "nn_ensemble_2_trainspecs_1"
model = keras.models.load_model(nn_dirname / "bestmodel.keras")
outputs, inputs = analysis.sensitivity_analysis(model, mode="random_normal")
output_main, input_main = analysis.sensitivity_analysis(model, resolution_1=1, mode=0.0)
analysis.plot_analysis(
    outputs,
    inputs,
    nn_dirname / "sensitivity_analysis",
    feature_names=list(FEATURE_TO_INDEX.keys()),
    main_plot=(output_main, input_main),
    legend=True,
    max_columns=3,
)
