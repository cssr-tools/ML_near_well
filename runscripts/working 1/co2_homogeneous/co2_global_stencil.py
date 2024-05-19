import os
from typing import Any

import numpy as np
import tensorflow as tf

from pyopmnearwell.ml import ensemble, integration, nn
from pyopmnearwell.utils import units

# Load the dataset and restructure the features.
dirname: str = os.path.dirname(__file__)
os.makedirs(os.path.join(dirname, "dataset_global_stencil"), exist_ok=True)
os.makedirs(os.path.join(dirname, "nn_global_stencil"), exist_ok=True)
os.makedirs(os.path.join(dirname, "integration_global_stencil"), exist_ok=True)

ds: tf.data.Dataset = tf.data.Dataset.load(os.path.join(dirname, "dataset"))
features, targets = next(iter(ds.batch(batch_size=len(ds)).as_numpy_iterator()))

# Reshape features and targets:
# Each feature has the shape ``(num_layers, 5)`` where the features in the second
# dimension are in the following order:
# 1. PRESSURE
# 2. SGAS
# 3. PERMX
# 4. PERMZ
# 5. TIME
# The targets have the shapw ``(num_layers)``.
num_layers: int = 5
ensemble.store_dataset(
    features.reshape(-1, num_layers, 5),
    targets.reshape(-1, num_layers),
    os.path.join(dirname, "dataset_global_stencil"),
)
# adjust to datasets including the upper and lower value
# Train model
# ``ninputs == num_features * numlayers``
model = nn.get_1D_CNN(noutputs=5)
train_data, val_data = nn.scale_and_prepare_dataset(
    os.path.join(dirname, "dataset_global_stencil"),
    feature_names=[
        "pressure",
        "saturation",
        "permx",
        "permz",
        "time",
    ],
    savepath=os.path.join(dirname, "nn_global_stencil"),
    conv_input=True,
)
nn.train(
    model,
    train_data,
    val_data,
    savepath=os.path.join(dirname, "nn_global_stencil"),
    epochs=100,
)
