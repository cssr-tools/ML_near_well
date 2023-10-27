"""Train a neural network that predicts the well index from the pressure, progressed
time, and distance from the well."""
from __future__ import annotations

import csv
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import runspecs
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

import pyopmnearwell.utils.units as units
from pyopmnearwell.ml.kerasify import export_model
from pyopmnearwell.utils.formulas import peaceman_WI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_RADII: int = 98

dirpath: str = os.path.dirname(os.path.realpath(__file__))
savepath: str = os.path.join(dirpath, "ml_model")
os.makedirs(savepath, exist_ok=True)
logdir: str = os.path.join(savepath, "logs")

# Load the entire dataset into a tensor for faster training.
logger.info("Load dataset.")
orig_ds = tf.data.Dataset.load(
    os.path.join(dirpath, "ensemble_runs", "pressure_radius_WI")
)
logger.info(
    f"Dataset at {os.path.join(dirpath, 'ensemble_runs', 'pressure_radius_WI')}"
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

# Write scaling to file
with open(os.path.join(savepath, "scales.csv"), "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["variable", "min", "max"])
    writer.writeheader()
    data_min = feature_scaler.data_min_
    data_max = feature_scaler.data_max_
    for feature_name, feature_min, feature_max in zip(
        [
            "pressure",
            "temperature",
            "permeability",
            "time",
            "radius",
        ],
        data_min,
        data_max,
    ):
        writer.writerow(
            {"variable": feature_name, "min": feature_min, "max": feature_max}
        )
    data_min = target_scaler.data_min_
    data_max = target_scaler.data_max_
    for feature_name, feature_min, feature_max in zip(["WI"], data_min, data_max):
        writer.writerow(
            {"variable": feature_name, "min": feature_min, "max": feature_max}
        )


# Shuffle once before splitting into training and val.
ds = orig_ds.shuffle(buffer_size=len(orig_ds))

# Split the dataset into a training and a validation data set.
train_size = int(0.9 * len(ds))
val_size = int(0.1 * len(ds))
train_ds = ds.take(train_size)
val_ds = ds.skip(train_size)

train_features, train_targets = next(
    iter(train_ds.batch(batch_size=len(train_ds)).as_numpy_iterator())
)
val_features, val_targets = next(
    iter(val_ds.batch(batch_size=len(val_ds)).as_numpy_iterator())
)
# Scale the features and targets.
train_features = feature_scaler.transform(train_features)
train_targets = target_scaler.transform(train_targets)
val_features = feature_scaler.transform(val_features)
val_targets = target_scaler.transform(val_targets)

logger.info(f"Scaled and transformed into training and validation dataset.")
# # Check the shape of the input and target tensors.
for x, y in train_ds:
    logger.info(f"shape of input tensor {x.shape}")
    logger.info(f"shape of output tensor {y.shape}")
    break


#  Create the neural network.
ninputs: int = 5
noutputs: int = 1
model = Sequential(
    [
        Input(shape=(ninputs,)),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(noutputs),
    ]
)


# Callbacks for model saving, learning rate decay and logging.
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(savepath, "bestmodel"),
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
)
lr_callback = (
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.1, patience=10, verbose=1, min_delta=1e-10, min_lr=1e-7
    ),
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# Train the model.
model.compile(
    loss="mse",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
)
model.fit(
    train_features,
    train_targets,
    batch_size=600,
    epochs=150,
    # Ignore Pylance complaining. This is an typing error in tensorflow/keras.
    verbose=1,  # type: ignore
    validation_data=(val_features, val_targets),
    callbacks=[checkpoint_callback, lr_callback, tensorboard_callback],
)
model.save_weights(os.path.join(savepath, "finalmodel"))

# Load the best model and save to OPM format.
model.load_weights(os.path.join(savepath, "bestmodel"))
export_model(model, os.path.join(savepath, "WI.model"))

# Plot the trained model vs. the data.
# Sample from the unshuffled data set to have the elements sorted.
features, targets = next(
    iter(orig_ds.batch(batch_size=len(orig_ds)).as_numpy_iterator())
)

# Loop through 3 ensemble members.
features = features.reshape((-1, NUM_RADII, ninputs))[:3, ...].reshape(
    -1, NUM_RADII, ninputs
)
targets = targets.reshape((-1, NUM_RADII, noutputs))[:3, ...].reshape(
    -1, NUM_RADII, noutputs
)

for i, (feature, target) in enumerate(list(zip(features, targets))):
    plt.figure()
    target_hat = target_scaler.inverse_transform(
        model(feature_scaler.transform(feature))
    )
    WI_analytical = (
        np.vectorize(peaceman_WI)(
            feature[0, 2] * units.MILIDARCY_TO_M2 * units.MILIDARCY_TO_M2,
            feature[..., 4],
            runspecs.WELL_RADIUS,
            runspecs.DENSITY,
            runspecs.VISCOSITY,
        )
        / runspecs.SURFACE_DENSITY
    )
    plt.plot(
        feature[..., 4],
        target_hat,
        label="nn",
    )
    plt.scatter(
        feature[::5, 4],
        target[::5],
        label="data",
    )
    plt.plot(
        feature[..., 4],
        WI_analytical,
        label="Peaceman",
    )
    plt.legend()
    plt.title(
        rf"$p={feature[-1][0]:.3e}\,[Pa]$ at $r={feature[-1][4]:.2f}\,[m]$"
        + "\n"
        + rf"$T={feature[0][1]:2f}\,[°C]$, $k={feature[0][2]:2f}\,[mD]$,"
        + rf"$t={feature[0][3]:2f}\,[h]$"
    )
    plt.xlabel(r"$r\,[m]$")
    plt.ylabel(r"$WI\,[m^4\cdot s/kg]$")
    plt.subplots_adjust(left=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f"nn_p_r_to_WI_{i}.png"))
    plt.show()
