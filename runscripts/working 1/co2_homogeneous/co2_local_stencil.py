import csv
import os
from typing import Any

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from pyopmnearwell.ml import ensemble, integration, nn
from pyopmnearwell.utils import formulas, units

height: float = 50
num_zcells: int = 10
npoints: int = 5
INJECTION_RATE_PER_SECOND: float = (
    5e4 * 6 * units.Q_per_day_to_Q_per_seconds
)  # unit: [m^3/s]
INIT_TEMPERATURE: float = 25  # unit: [°C])
PERMX: float = 1e-12 * units.M2_TO_MILIDARCY  # unit: [mD]
SURFACE_DENSITY: float = 1.86843  # unit: [kg/m^3]
timesteps: np.ndarray = np.arange(34)  # No unit.


# Load the dataset and restructure the features.
dirname: str = os.path.dirname(__file__)
data_dirname: str = os.path.join(dirname, "dataset_averaged_pressure_local_stencil")
nn_dirname: str = os.path.join(dirname, "nn_local_averaged_pressure_stencil")
os.makedirs(data_dirname, exist_ok=True)
os.makedirs(nn_dirname, exist_ok=True)
os.makedirs(
    os.path.join(dirname, "integration_averaged_pressure_local_stencil"), exist_ok=True
)

ds: tf.data.Dataset = tf.data.Dataset.load(
    os.path.join(dirname, "dataset_averaged_pressure")
)
features, targets = next(iter(ds.batch(batch_size=len(ds)).as_numpy_iterator()))

# At the upper and lower boundary the neighbor values are padded with zeros. Note that
# the arrays go from upper to lower cells.
new_features: list[np.ndarray] = []
for i in range(features.shape[-1] - 1):
    feature = features[..., i]
    # The features have ``ndim == 3``: ``(npoints, num_timesteps, num_layers)``. The
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
# 7. TIME

ensemble.store_dataset(
    np.stack(new_features, axis=-1),
    targets.flatten()[..., None],
    data_dirname,
)

# Train model
model = nn.get_FCNN(ninputs=7, noutputs=1)
train_data, val_data = nn.scale_and_prepare_dataset(
    data_dirname,
    feature_names=[
        "pressure_upper",
        "pressure",
        "pressure_lower",
        "saturation_upper",
        "saturation",
        "saturation_lower",
        "time",
    ],
    savepath=nn_dirname,
)
nn.train(
    model,
    train_data,
    val_data,
    savepath=nn_dirname,
    epochs=100,
)


# Plot the nn

for feature_member, target_member, i in list(
    zip(
        new_features[2][:: features.shape[0], ..., 0, :],
        targets[:: features.shape[0], ..., 0],
        range(features[:: features.shape[0], ..., 0, 0].shape[0]),
    )
):
    pressure_member: np.ndarray = feature_member[..., 1]
    saturation_member: np.ndarray = feature_member[..., 4]

    # Loop through all time steps and collect analytical WIs.
    WI_analytical = []
    for j in range(pressure_member.shape[0]):
        pressure = pressure_member[j]
        saturation = saturation_member[j]
        # Evalute density and viscosity.
        densities = []
        viscosities = []
        for phase in ["CO2", "water"]:
            densities.append(
                formulas.co2brinepvt(
                    pressure=pressure,
                    temperature=INIT_TEMPERATURE + units.CELSIUS_TO_KELVIN,
                    property="density",
                    phase=phase,
                )
            )
            viscosities.append(
                formulas.co2brinepvt(
                    pressure=pressure,
                    temperature=INIT_TEMPERATURE + units.CELSIUS_TO_KELVIN,
                    property="viscosity",
                    phase=phase,
                )
            )
        # Calculate the well index from two-phase Peaceman. Note that the relative
        # permeabilty functions are quadratic. The analytical well index is in [m*s],
        # hence we need to devide by density to transform to [m^4*s/kg].
        WI_analytical.append(
            formulas.two_phase_peaceman_WI(
                k_h=PERMX * units.MILIDARCY_TO_M2 * height / num_zcells,
                r_e=formulas.equivalent_well_block_radius(100),
                r_w=0.25,
                rho_1=densities[0],
                mu_1=viscosities[0],
                k_r1=saturation**2,
                rho_2=densities[1],
                mu_2=viscosities[1],
                k_r2=(1 - saturation) ** 2,
            )
            / SURFACE_DENSITY
        )
    # Compute total mobility
    injection_rate: float = INJECTION_RATE_PER_SECOND
    WI_nn: np.ndarray = nn.scale_and_evaluate(
        model, features, os.path.join(nn_dirname, "scalings.csv")
    )
    bhp_nn: np.ndarray = injection_rate / WI_nn + pressure
    bhp_data: np.ndarray = injection_rate / target_member + pressure
    bhp_analytical: np.ndarray = injection_rate / np.array(WI_analytical) + pressure
    plt.figure()
    plt.scatter(
        timesteps * 3 + 1,
        bhp_data,
        label="data",
    )
    plt.plot(
        timesteps * 3 + 1,
        bhp_analytical,
        label="NN",
    )
    plt.plot(
        timesteps * 3 + 1,
        bhp_analytical,
        label="two-phase Peaceman",
    )
    plt.legend()
    plt.xlabel(r"$t\,[d]$")
    plt.ylabel(r"$p\,[Pa]$")
    plt.savefig(
        os.path.join(
            dirname,
            "nn_local_averaged_pressure_stencil",
            f"p_data_vs_nn_vs_Peaceman_{i}.png",
        )
    )
    plt.show()
