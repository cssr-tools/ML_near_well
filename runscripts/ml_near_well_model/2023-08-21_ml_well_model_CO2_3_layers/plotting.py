"""Plot some nn and data things."""
from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

import pyopmnearwell.utils.units as units

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_REPORT_STEPS: int = 34
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
full_features, targets = next(
    iter(orig_ds.batch(batch_size=len(orig_ds)).as_numpy_iterator())
)
logger.info("Adapt MinMaxScalers")
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
feature_scaler.fit(full_features)
target_scaler.fit(targets)

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

# Load the best model and save to OPM format.
model.load_weights(os.path.join(savepath, "bestmodel"))

# Loop through 5 ensemble members. Fix a radius and plot model vs data w.r.t. the time.
features = full_features.reshape((-1, NUM_REPORT_STEPS, NUM_RADII, ninputs))[
    :20, :, 20, :
]
targets = targets.reshape((-1, NUM_REPORT_STEPS, NUM_RADII, noutputs))[:20, :, 20, :]
bhps = full_features.reshape((-1, NUM_REPORT_STEPS, NUM_RADII, ninputs))[:20, :, 0, 0]

for i, (feature, target, bhp) in enumerate(list(zip(features, targets, bhps))):
    target_hat = target_scaler.inverse_transform(
        model(feature_scaler.transform(feature))
    )

    # plt.figure()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(
        feature[..., 3],
        target_hat,
        label="nn",
    )
    ax1.scatter(
        feature[::5, 3],
        target[::5],
        label="data",
    )
    ax1.set_xlabel(r"$t\,[h]$")
    ax1.set_ylabel(r"$WI\,[m^4\cdot s/kg]$")

    # ax2.plot(
    #     feature[..., 3],
    #     (bhp - feature[..., 0]) * units.PASCAL_TO_BAR,
    #     label="data",
    # )
    ax2.plot(
        feature[..., 3],
        feature[..., 0] * units.PASCAL_TO_BAR,
        label=r"data $p_{gb}$",
    )
    ax2.plot(
        feature[..., 3],
        bhp * units.PASCAL_TO_BAR,
        label=r"data $p_{bh}$",
    )
    # ax2.plot(
    #     feature[..., 3],
    #     (bhp - feature[..., 0]) * units.PASCAL_TO_BAR,
    #     label=r"data $\Delta p$",
    # )

    ax2.set_ylabel(r"p\,[bar]$")

    fig.legend()
    ax1.set_title(
        rf"$p={feature[-1][0]:.3e}\,[Pa]$ at $t={feature[-1][3]:.2f}\,[h]$"
        + "\n"
        + rf"$T={feature[0][1]:2f}\,[°C]$, $k={feature[0][2]:2f}\,[mD]$,"
        + rf"$r={feature[0][4]:2f}\,[m]$"
    )
    fig.tight_layout()
    plt.savefig(os.path.join(savepath, f"nn_vs_data_along_time_{i}.png"))
    # plt.show()
