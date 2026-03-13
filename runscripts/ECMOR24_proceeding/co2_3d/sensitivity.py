import pathlib
import keras

from pyopmnearwell.ml.analysis import sensitivity_analysis, plot_analysis

# Juster disse om modellen ligger
from runspecs import trainspecs   # <-- legg til
NN_DIR = pathlib.Path("nn")
MODEL_PATH = NN_DIR / "bestmodel.keras"   # noen ganger heter den best_model.keras

FEATURE_NAMES = [
    "pressure_upper",
    "pressure",
    "pressure_lower",
    "saturation_upper",
    "saturation",
    "saturation_lower",
    "radius",
    "total_injected_volume",
    "injection_rate",
    "time_days",
    "PI_analytical",
]

def main():
    model = keras.models.load_model(MODEL_PATH)

    outputs, inputs = sensitivity_analysis(
        model,
        resolution_1=15,   # antall ulike "konstante settinger" -> antall linjer per subplot
        resolution_2=80,   # punkter langs x-aksen
        mode="random_uniform" # start enkelt; du kan også prøve "random_uniform"
    )

    outdir = NN_DIR / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    plot_analysis(
        outputs,
        inputs,
        savepath=outdir / "sensitivity_fcnn",
        feature_names=trainspecs["features"],  # <-- viktig
        legend=False,
        max_columns=3,
    )

if __name__ == "__main__":
    main()
