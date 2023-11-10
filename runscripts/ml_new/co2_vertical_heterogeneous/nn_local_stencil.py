import os

import keras_tuner
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from pyopmnearwell.ml import ensemble, nn
from pyopmnearwell.utils import formulas, units
from runspecs import runspecs_ensemble_1 as runspecs_ensemble
from runspecs import trainspecs_1 as trainspecs
from tensorflow import keras

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
    "total_injected_volume": 9,
    "WI_analytical": 10,
}


# Create dirs
dirname: str = os.path.dirname(__file__)

data_dirname: str = os.path.join(dirname, f"dataset_{runspecs_ensemble['name']}")
new_data_dirname: str = os.path.join(
    dirname, f"dataset_{runspecs_ensemble['name']}_{trainspecs['name']}"
)
os.makedirs(new_data_dirname, exist_ok=True)

nn_dirname: str = os.path.join(
    dirname, f"nn_{runspecs_ensemble['name']}_{trainspecs['name']}"
)
os.makedirs(nn_dirname, exist_ok=True)

# Restructure data:
ds: tf.data.Dataset = tf.data.Dataset.load(data_dirname)
features, targets = next(iter(ds.batch(batch_size=len(ds)).as_numpy_iterator()))

# Pad values at the upper and lower boundary.
# NOTE: The arrays range FROM upper TO lower cells.
new_features: list[np.ndarray] = []
for i in range(features.shape[-1] - 1):
    feature: np.ndarray = features[..., i]

    # Pressure options:
    if i == 0:
        if trainspecs["pressure_unit"] == "bar":
            feature = feature * units.PASCAL_TO_BAR

        if trainspecs["pressure_padding"] == "zeros":
            padding_mode: str = "constant"
            padding_value: float = 0.0

        # TODO: Fix init padding mode. Where to get the pressure value from? The
        # runspecs only have the ensemble values. The data truncates the init value.
        # elif trainspecs["pressure_padding"] == "init":
        #     padding_mode = "constant"
        #     padding_values  = runspecs_ensemble["constant"]

        elif trainspecs["pressure_padding"] == "neighbor":
            padding_mode = "edge"
            padding_value = 0.0

    # Permeability options:
    elif i == 2:
        if trainspecs["permeability_log"]:
            feature = np.log10(feature)

        if trainspecs["permeability_padding"] == "zeros":
            padding_mode = "constant"
            padding_value = 0.0

    # The features have ``ndim == 3``: ``(npoints, num_timesteps, num_layers)``. The
    # last dimension is padded.
    # Ignore MypY complaining.
    if padding_mode == "constant":
        feature_upper = np.pad(  # type: ignore
            feature[..., :-1],
            ((0, 0), (0, 0), (1, 0)),
            mode=padding_mode,
            constant_values=padding_value,
        )
        feature_lower = np.pad(  # type: ignore
            feature[..., 1:],
            ((0, 0), (0, 0), (0, 1)),
            mode=padding_mode,
            constant_values=padding_value,
        )
    else:
        feature_upper = np.pad(  # type: ignore
            feature[..., :-1],
            ((0, 0), (0, 0), (1, 0)),
            mode=padding_mode,
        )
        feature_lower = np.pad(  # type: ignore
            feature[..., 1:],
            ((0, 0), (0, 0), (0, 1)),
            mode=padding_mode,
        )

    new_features.extend([feature_upper, feature, feature_lower])

# Total injected volume and WI_analytical were not padded and are added back again.
new_features.append(features[..., -2])

if trainspecs["WI_log"]:
    targets = np.log10(targets)
    new_features.append(np.log10(features[..., -1]))
else:
    new_features.append(features[..., -1])

new_features_array: np.ndarray = np.stack(new_features, axis=-1)

# Select the correct features from the train specs
new_features_final: np.ndarray = new_features_array[
    ..., [FEATURE_TO_INDEX[feature] for feature in trainspecs["features"]]
]

# The new features are in the following order:
# 1. PRESSURE - upper neighbor
# 2. PRESSURE - cell
# 3. PRESSURE - lower neighbor
# 4. SATURATION - upper neighbor
# 5. SATURATION - cell
# 6. SATURATION - lower neighbor
# 4. PERMEABILITY - upper neighbor
# 5. PERMEABILITY - cell
# 6. PERMEABILITY - lower neighbor
# 7. total injected gas
# . analytical WI

ensemble.store_dataset(
    new_features_final.reshape(-1, new_features_final.shape[-1]),
    targets.flatten()[..., None],
    new_data_dirname,
)


# Train model:
# model = nn.get_FCNN(
#     ninputs=len(trainspecs["features"]),
#     noutputs=1,
#     depth=trainspecs["depth"],
#     hidden_dim=trainspecs["hidden_dim"],
#     kernel_initializer="glorot_uniform",
#     normalization=trainspecs["Z-normalization"],
# )
train_data, val_data = nn.scale_and_prepare_dataset(
    new_data_dirname,
    feature_names=trainspecs["features"],
    savepath=nn_dirname,
    scale=trainspecs["MinMax_scaling"],
)

train_features, train_targets = train_data
assert not np.any(np.isnan(train_features))
assert not np.any(np.isnan(train_targets))
val_features, val_targets = val_data
assert not np.any(np.isnan(train_features))
assert not np.any(np.isnan(val_targets))

# # Adapt the layers when using z-normalization.
# if trainspecs["Z-normalization"]:
#     model.layers[0].adapt(train_data[0])
#     model.layers[-1].adapt(train_data[1])
if "percentage_loss" in trainspecs and trainspecs["percentage_loss"]:
    sample_weight: np.ndarray = 1 / (np.abs(train_targets) + np.finfo(float).eps)
else:
    sample_weight = np.ones_like(train_targets)

kerasify = not trainspecs["Z-normalization"]
model: keras.Model = nn.tune(
    len(trainspecs["features"]), 1, train_data, val_data, sample_weight=sample_weight
)
nn.train(
    model,
    train_data,
    val_data,
    nn_dirname,
    recompile_model=False,
    sample_weight=sample_weight,
)


############
# Plotting #
############
# Comparison nn WI vs. Peaceman WI vs. data WI for 3 layers for the first ensemble
# member.
timesteps: np.ndarray = np.linspace(0, 1, features.shape[-3]) / 1  # unit: [day]

fig = plt.figure()
for i, color in zip([0, 2, 4], plt.cm.rainbow(np.linspace(0, 1, 3))):
    features: np.ndarray = new_features_final[0, ..., i, :]

    # Cell pressure is given by the 2nd feature.
    pressures: np.ndarray = new_features_array[0, ..., i, 1]
    WI_analytical: np.ndarray = new_features_array[0, ..., i, -1]
    WI_data: np.ndarray = targets[0, ..., i, 0]
    WI_nn: np.ndarray = nn.scale_and_evaluate(
        model, features, os.path.join(nn_dirname, "scalings.csv")
    )[..., 0]

    if trainspecs["WI_log"]:
        WI_analytical = 10**WI_analytical
        WI_data = 10**WI_data
        WI_nn = 10**WI_nn

    # Plot analytical vs. data WI in the upper layer.
    plt.scatter(timesteps, WI_data, label=f"Layer {i} data", color=color, linestyle="-")
    plt.plot(
        timesteps,
        WI_analytical,
        label=f"Layer {i}: Peaceman",
        color=color,
        linestyle="--",
    )
    plt.plot(timesteps, WI_nn, label=f"Layer {i}: NN", color=color, linestyle="-")

plt.legend()
plt.xlabel(r"$t\,[d]$")
plt.ylabel(r"$WI\,[m^4\cdot s/kg]$")
plt.title(f"WI")
plt.savefig(os.path.join(dirname, nn_dirname, "WI_data_vs_nn_vs_Peaceman.png"))
plt.show()
