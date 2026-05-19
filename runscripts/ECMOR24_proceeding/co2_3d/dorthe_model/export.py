"""Export the trained neural network model using kerasify."""

import json
import pathlib

from pyopmnearwell.ml.kerasify import export_model
from tensorflow import keras


def main():
    """Load the model and export with kerasify."""
    # Define paths
    model_dir = pathlib.Path(__file__).parent
    config_file = model_dir / "MLNearWellConfig.json"

    # Load the keras model
    model = keras.models.load_model(model_dir / "bestmodel.keras")

    # Export the model using kerasify
    output_model_path = model_dir / "WI.model"
    export_model(model, output_model_path)

    # Update the config file with the model path
    with config_file.open("r", newline="", encoding="utf-8") as f:
        config = json.load(f)

    config["model_path"] = str(output_model_path)

    with config_file.open("w", newline="", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print(f"Model exported to {output_model_path}")
    print(f"Config updated: {config_file}")


if __name__ == "__main__":
    main()
