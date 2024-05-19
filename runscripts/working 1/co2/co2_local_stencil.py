import os
from typing import Any

import numpy as np
import tensorflow as tf

from pyopmnearwell.ml import ensemble, integration, nn
from pyopmnearwell.utils import units

# Load the dataset and restructure the features.
dirname: str = os.path.dirname(__file__)
os.makedirs(os.path.join(dirname, "dataset_local_stencil"), exist_ok=True)
os.makedirs(os.path.join(dirname, "nn_local_stencil"), exist_ok=True)
os.makedirs(os.path.join(dirname, "integration_local_stencil"), exist_ok=True)

ds: tf.data.Dataset = tf.data.Dataset.load(os.path.join(dirname, "dataset"))
features, targets = next(iter(ds.batch(batch_size=len(ds)).as_numpy_iterator()))

# At the upper and lower boundary the neighbor values are padded with zeros. Note that
# the arrays go from upper to lower cells.
new_features: list[np.ndarray] = []
for i in range(features.shape[-1] - 1):
    feature = features[..., i]
    # The features have ``ndim == 3``: ``(num_members, num_timesteps, num_cells)``. The
    # last dimension is padded.
    feature_upper = np.pad(
        feature[..., :-1],
        ((0, 0), (0, 0), (1, 0)),
        mode="constant",
        constant_values=0,
    )
    feature_lower = np.pad(
        feature[..., 1:],
        ((0, 0), (0, 0), (0, 1)),
        mode="constant",
        constant_values=0,
    )
    new_features.extend(
        [feature_upper.flatten(), feature.flatten(), feature_lower.flatten()]
    )

# Add time back again.
new_features.append(features[..., -1].flatten())

# The new features are in the following order:
# 1. PRESSURE - upper neighbor
# 2. PRESSURE - cell
# 3. PRESSURE - lower neighbor
# 4. SGAS - upper neighbor
# 5. SGAS - cell
# 6. SGAS - lower neighbor
# 7. PERMX - upper neighbor
# 8. PERMX - cell
# 9. PERMX - lower neighbor
# 10. PERMZ - upper neighbor
# 11. PERMZ - cell
# 12. PERMZ - lower neighbor
# 13. TIME

ensemble.store_dataset(
    np.stack(new_features, axis=-1),
    targets.flatten()[..., None],
    os.path.join(dirname, "dataset_local_stencil"),
)
# adjust to datasets including the upper and lower value
# Train model
# ``ninputs == num_features * numlayers``
model = nn.get_FCNN(ninputs=13, noutputs=1)
train_data, val_data = nn.scale_and_prepare_dataset(
    os.path.join(dirname, "dataset_local_stencil"),
    feature_names=[
        "pressure_upper",
        "pressure",
        "pressure_lower",
        "saturation_upper",
        "saturation",
        "saturation_lower",
        "permx_upper",
        "permx",
        "permx_lower",
        "permz_upper",
        "permz",
        "permz_lower",
        "time",
    ],
    savepath=os.path.join(dirname, "nn_local_stencil"),
)
nn.train(
    model,
    train_data,
    val_data,
    savepath=os.path.join(dirname, "nn_local_stencil"),
    epochs=100,
)
