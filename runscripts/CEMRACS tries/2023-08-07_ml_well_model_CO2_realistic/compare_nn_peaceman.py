"""Train a neural network that predicts the well index from the permeability, initital
reservoir pressure and distance from the well."""

import csv
import logging
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

from pyopmnearwell.ml.kerasify import export_model
from pyopmnearwell.utils.formulas import peaceman_WI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dirpath: str = os.path.dirname(os.path.realpath(__file__))
savepath: str = os.path.join(dirpath, "model_permeability_radius_WI")

# Load the entire dataset into a tensor.
logger.info("Load dataset.")
orig_ds = tf.data.Dataset.load(
    os.path.join(dirpath, "ensemble_runs", "permeability_radius_WI")
)
logger.info(
    f"Dataset at {os.path.join(dirpath, 'ensemble_runs', 'permeability_radius_WI')}"
    + f" contains {len(orig_ds)} samples."
)

# Adapt the input & output scaling.
features, targets = next(
    iter(orig_ds.batch(batch_size=len(orig_ds)).as_numpy_iterator())
)
logger.info("Adapt MinMaxScalers")
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
feature_scaler.fit(features)
target_scaler.fit(targets)

# Load the best model.
model = Sequential(
    [
        Input(shape=(3,)),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(1),
    ]
)
model.load_weights(os.path.join(savepath, "bestmodel"))

# Plot the trained model vs. the data vs. Peaceman.
# Sample from the unshuffled data set to have the elements sorted.
features, targets = next(
    iter(orig_ds.batch(batch_size=len(orig_ds)).as_numpy_iterator())
)

# PLot for varying permeabilities and fixed initial pressure.
plt.figure()
# Reshape into the number of different permeabilities, initial pressures and radii,
# to extract datapoints for a fixed pressure.
features_fixed_init_pressure = features.reshape((30, 20, 395, 3))[::10, 0, ...]
targets_fixed_init_pressure = targets.reshape((30, 20, 395, 1))[::10, 0, ...]
peaceman_fixed_init_pressure = peaceman_WI(
    features_fixed_init_pressure[..., 0], features_fixed_init_pressure[..., 2], 0.12
)

for feature, target, peaceman in zip(
    features_fixed_init_pressure,
    targets_fixed_init_pressure,
    peaceman_fixed_init_pressure,
):
    target_hat: tf.Tensor = target_scaler.inverse_transform(
        model(feature_scaler.transform(feature.reshape((395, 3))))
    )
    k = feature[0][0]
    plt.plot(
        feature[..., 2].flatten(),
        tf.reshape(target_hat, (-1)),
        label=rf"$k={k}$ nn [m^2]",
    )
    plt.plot(
        feature[..., 2].flatten(),
        tf.reshape(peaceman, (-1)),
        label=rf"$k={k}$ Peaceman [m^2]",
    )
    plt.scatter(
        feature[..., 2].flatten()[::5], target.flatten()[::5], label=rf"$k={k}$ s [m^2]"
    )
plt.legend()
plt.xlabel(r"$r$ [m]")
plt.ylabel(r"$WI$ [Nm/s]")
plt.title(rf"initial reservoir pressure ${feature[0][1]}$ [bar]")
plt.savefig(os.path.join(savepath, "nn_k_r_to_WI_1.png"))
plt.show()


# Different pressure.
# features_fixed_init_pressure = features.reshape((30, 20, 395, 3))[::10, 10, ...]
# targets_fixed_init_pressure = targets.reshape((30, 20, 395, 1))[::10, 10, ...]
# plt.figure()
# for feature, target in zip(features_fixed_init_pressure, targets_fixed_init_pressure):
#     target_hat = target_scaler.inverse_transform(
#         model(feature_scaler.transform(feature.reshape((395, 3))))
#     )
#     k = feature[0][0]
#     plt.plot(
#         feature[..., 2].flatten(),
#         tf.reshape(target_hat, (-1)),
#         label=rf"$k={k}$ nn [m^2]",
#     )
#     plt.scatter(
#         feature[..., 2].flatten()[::5], target.flatten()[::5], label=rf"$k={k}$ s [m^2]"
#     )
# plt.legend()
# plt.xlabel(r"$r$ [m]")
# plt.ylabel(r"$WI$ [Nm/s]")
# plt.title(rf"initial reservoir pressure ${feature[0][1]}$ [bar]")
# plt.savefig(os.path.join(savepath, "nn_k_r_to_WI_2.png"))
# plt.show()

# # Extract datapoints for fixed permeability.
# features_fixed_permeability = features.reshape((30, 20, 395, 3))[2, ::3, ...]
# targets_fixed_permeability = targets.reshape((30, 20, 395, 1))[2, ::3, ...]
# plt.figure
# for feature, target in zip(features_fixed_permeability, targets_fixed_permeability):
#     target_hat = target_scaler.inverse_transform(
#         model(feature_scaler.transform(feature.reshape((395, 3))))
#     )
#     p = feature[0][1]
#     plt.plot(
#         feature[..., 2].flatten(),
#         tf.reshape(target_hat, (-1)),
#         label=rf"$p_i={p}$ nn [m^2]",
#     )
#     plt.scatter(
#         feature[..., 2].flatten()[::5],
#         target.flatten()[::5],
#         label=rf"$p_i={p}$ s [m^2]",
#     )
# plt.legend()
# plt.xlabel(r"$r[m]$")
# plt.ylabel(r"$WI$ [Nm/s]")
# plt.title(rf"permeability ${feature[0][0]} [m^2]$")
# plt.savefig(os.path.join(savepath, "nn_p_r_to_WI_1.png"))
# plt.show()

# # Different fixed permeability.
# features_fixed_permeability = features.reshape((30, 20, 395, 3))[15, ::3, ...]
# targets_fixed_permeability = targets.reshape((30, 20, 395, 1))[15, ::3, ...]
# plt.figure
# for feature, target in zip(features_fixed_permeability, targets_fixed_permeability):
#     target_hat = target_scaler.inverse_transform(
#         model(feature_scaler.transform(feature.reshape((395, 3))))
#     )
#     p = feature[0][1]
#     plt.plot(
#         feature[..., 2].flatten(),
#         tf.reshape(target_hat, (-1)),
#         label=rf"$p_i={p}$ nn [m^2]",
#     )
#     plt.scatter(
#         feature[..., 2].flatten()[::5],
#         target.flatten()[::5],
#         label=rf"$p_i={p}$ s [m^2]",
#     )
# plt.legend()
# plt.xlabel(r"$r[m]$")
# plt.ylabel(r"$WI$ [Nm/s]")
# plt.title(rf"permeability ${feature[0][0]} [m^2]$")
# plt.savefig(os.path.join(savepath, "nn_p_r_to_WI_2.png"))
# plt.show()
