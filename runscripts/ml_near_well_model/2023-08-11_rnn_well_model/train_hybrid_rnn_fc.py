"""Train a neural network that predicts the well index from the permeability, initital
reservoir pressure and distance from the well, as well as a time series condsisting of
the injection schedule."""

import csv
import logging
import os

import keras
import matplotlib.pyplot as plt
import tensorflow as tf

# from kerasify import export_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Concatenate, Dense, Input, SimpleRNN
from tensorflow.keras.models import Model, Sequential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dirpath: str = os.path.dirname(os.path.realpath(__file__))
savepath: str = os.path.join(dirpath, "model_permeability_radius_WI")
os.makedirs(savepath, exist_ok=True)
logdir: str = os.path.join(savepath, "logs")

# # Load the entire dataset into a tensor for faster training.
# logger.info("Load dataset.")
# orig_ds = tf.data.Dataset.load(
#     os.path.join(dirpath, "ensemble_runs", "permeability_radius_WI")
# )
# logger.info(
#     f"Dataset at {os.path.join(dirpath, 'ensemble_runs', 'permeability_radius_WI')}"
#     + f" contains {len(orig_ds)} samples."
# )
# # Adapt the input & output scaling.
# features, targets = next(
#     iter(orig_ds.batch(batch_size=len(orig_ds)).as_numpy_iterator())
# )
# logger.info("Adapt MinMaxScalers")
# feature_scaler = MinMaxScaler()
# target_scaler = MinMaxScaler()
# feature_scaler.fit(features)
# target_scaler.fit(targets)

# # Write scaling to file
# with open(os.path.join(savepath, "scales.csv"), "w", newline="") as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=["variable", "min", "max"])
#     writer.writeheader()
#     data_min = feature_scaler.data_min_
#     data_max = feature_scaler.data_max_
#     for feature_name, feature_min, feature_max in zip(
#         ["permeability", "init_pressure", "radius"], data_min, data_max
#     ):
#         writer.writerow(
#             {"variable": feature_name, "min": feature_min, "max": feature_max}
#         )
#     data_min = target_scaler.data_min_
#     data_max = target_scaler.data_max_
#     for feature_name, feature_min, feature_max in zip(["WI"], data_min, data_max):
#         writer.writerow(
#             {"variable": feature_name, "min": feature_min, "max": feature_max}
#         )


# # Shuffle once before splitting into training and val.
# ds = orig_ds.shuffle(buffer_size=len(orig_ds))

# # Split the dataset into a training and a validation data set.
# train_size = int(0.9 * len(ds))
# val_size = int(0.1 * len(ds))
# train_ds = ds.take(train_size)
# val_ds = ds.skip(train_size)

# train_features, train_targets = next(
#     iter(train_ds.batch(batch_size=len(train_ds)).as_numpy_iterator())
# )
# val_features, val_targets = next(
#     iter(val_ds.batch(batch_size=len(val_ds)).as_numpy_iterator())
# )
# # Scale the targets. The feature scaling is done inside the model.
# train_targets = target_scaler.transform(train_targets)
# val_targets = target_scaler.transform(val_targets)

# logger.info(f"Scaled and transformed into training and validation dataset.")
# # # Check the shape of the input and target tensors.
# for x, y in train_ds:
#     logger.info(f"shape of input tensor {x.shape}")
#     logger.info(f"shape of output tensor {y.shape}")
#     break


# Create the neural network.
# The RNN backbone treats the sequential features (injection rates and times), while the
# fully connected backbone treats the constant features (permeability, pressure,
# radius). The output of the RNN and the output of the FFNN are merged and a head
# predicts the well index.
sequential_input = Input(shape=(None, 2))
constant_input = Input(shape=(3,))
input = [sequential_input, constant_input]
backbone1 = SimpleRNN(10)(sequential_input)
backbone2 = Sequential(
    [
        constant_input,
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
        Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"),
    ]
)

merged_input = Concatenate()([backbone1, backbone2.output])
head = Dense(10, activation="sigmoid", kernel_initializer="glorot_normal")(merged_input)
head = Dense(10, activation="sigmoid", kernel_initializer="glorot_normal")(head)
head = Dense(10, activation="sigmoid", kernel_initializer="glorot_normal")(head)
head = Dense(10, activation="sigmoid", kernel_initializer="glorot_normal")(head)
head = Dense(10, activation="sigmoid", kernel_initializer="glorot_normal")(head)
head = Dense(1, activation="sigmoid", kernel_initializer="glorot_normal")(head)

model = Model(input, head)
logger.info(model.summary())
# Callbacks for model saving, learning rate decay and logging.
# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     os.path.join(savepath, "bestmodel"),
#     monitor="val_loss",
#     verbose=1,
#     save_best_only=True,
#     save_weights_only=True,
# )
# lr_callback = (
#     tf.keras.callbacks.ReduceLROnPlateau(
#         monitor="loss", factor=0.1, patience=10, verbose=1, min_delta=1e-10
#     ),
# )
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# # Train the model.
# model.compile(
#     loss="mse",
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
# )
# model.fit(
#     train_features,
#     target_scaler.transform(train_targets),
#     batch_size=600,
#     epochs=20,
#     # Ignore Pylance complaining. This is an typing error in tensorflow/keras.
#     verbose=1,  # type: ignore
#     validation_data=(val_features, target_scaler.transform(val_targets)),
#     callbacks=[checkpoint_callback, lr_callback, tensorboard_callback],
# )

# # Save to OPM
# # export_model(model)

# # Plot the trained model vs. the data.
# # Sample from the unshuffled data set to have the elements sorted.
# features, targets = next(
#     iter(orig_ds.batch(batch_size=len(orig_ds)).as_numpy_iterator())
# )

# # PLot for varying permeabilities and fixed initial pressure.
# plt.figure()
# # Reshape into the number of different permeabilities, initial pressures and radii,
# # to extract datapoints for a fixed pressure.
# features_fixed_init_pressure = features.reshape((30, 20, 395, 3))[::10, 0, ...]
# targets_fixed_init_pressure = targets.reshape((30, 20, 395, 1))[::10, 0, ...]
# for feature, target in zip(features_fixed_init_pressure, targets_fixed_init_pressure):
#     target_hat: tf.Tensor = target_scaler.inverse_transform(
#         model(feature.reshape((395, 3)))
#     )
#     k = feature[0][0]
#     plt.plot(
#         feature[..., 2].flatten(),
#         tf.reshape(target_hat, (-1)),
#         label=rf"$k={k}$ nn",
#     )
#     plt.scatter(
#         feature[..., 2].flatten()[::5], target.flatten()[::5], label=rf"$k={k}$ s"
#     )
# plt.legend()
# plt.xlabel(r"$r[m]$")
# plt.ylabel(r"$WI[kg/...]$")
# plt.title(rf"initial reservoir pressure ${feature[0][1]}$")
# plt.savefig(os.path.join(savepath, "hybrid_k_r_to_WI_1.png"))
# plt.show()


# # Different pressure.
# features_fixed_init_pressure = features.reshape((30, 20, 395, 3))[::10, 10, ...]
# targets_fixed_init_pressure = targets.reshape((30, 20, 395, 1))[::10, 10, ...]
# plt.figure()
# for feature, target in zip(features_fixed_init_pressure, targets_fixed_init_pressure):
#     target_hat = target_scaler.inverse_transform(model(feature.reshape((395, 3))))
#     k = feature[0][0]
#     plt.plot(
#         feature[..., 2].flatten(),
#         tf.reshape(target_hat, (-1)),
#         label=rf"$k={k}$ nn",
#     )
#     plt.scatter(
#         feature[..., 2].flatten()[::5], target.flatten()[::5], label=rf"$k={k}$ s"
#     )
# plt.legend()
# plt.xlabel(r"$r[m]$")
# plt.ylabel(r"$WI[kg/...]$")
# plt.title(rf"initial reservoir pressure ${feature[0][1]}$")
# plt.savefig(os.path.join(savepath, "hybrid_k_r_to_WI_2.png"))
# plt.show()

# # Extract datapoints for fixed permeability.
# features_fixed_permeability = features.reshape((30, 20, 395, 3))[2, ::3, ...]
# targets_fixed_permeability = targets.reshape((30, 20, 395, 1))[2, ::3, ...]
# plt.figure
# for feature, target in zip(features_fixed_permeability, targets_fixed_permeability):
#     target_hat = target_scaler.inverse_transform(model(feature.reshape((395, 3))))
#     p = feature[0][1]
#     plt.plot(
#         feature[..., 2].flatten(),
#         tf.reshape(target_hat, (-1)),
#         label=rf"$p_i={p}$ nn",
#     )
#     plt.scatter(
#         feature[..., 2].flatten()[::5], target.flatten()[::5], label=rf"$p_i={p}$ s"
#     )
# plt.legend()
# plt.xlabel(r"$r[m]$")
# plt.ylabel(r"$WI[kg/...]$")
# plt.title(rf"permeability ${feature[0][0]}$")
# plt.savefig(os.path.join(savepath, "hybrid_p_r_to_WI_1.png"))
# plt.show()

# # Different fixed permeability.
# features_fixed_permeability = features.reshape((30, 20, 395, 3))[15, ::3, ...]
# targets_fixed_permeability = targets.reshape((30, 20, 395, 1))[15, ::3, ...]
# plt.figure
# for feature, target in zip(features_fixed_permeability, targets_fixed_permeability):
#     target_hat = target_scaler.inverse_transform(model(feature.reshape((395, 3))))
#     p = feature[0][1]
#     plt.plot(
#         feature[..., 2].flatten(),
#         tf.reshape(target_hat, (-1)),
#         label=rf"$p_i={p}$ nn",
#     )
#     plt.scatter(
#         feature[..., 2].flatten()[::5], target.flatten()[::5], label=rf"$p_i={p}$ s"
#     )
# plt.legend()
# plt.xlabel(r"$r[m]$")
# plt.ylabel(r"$WI[kg/...]$")
# plt.title(rf"permeability ${feature[0][0]}$")
# plt.savefig(os.path.join(savepath, "hybrid_p_r_to_WI_2.png"))
# plt.show()
