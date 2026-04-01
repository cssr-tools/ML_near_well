from __future__ import annotations

import math
from typing import Any
import numpy as np

from upscale import CO2_3D_upscaler


class CO2_3D_upscaler_extrapolation(CO2_3D_upscaler):
    def __init__(
        self,
        data: np.ndarray,
        runspecs: dict[str, Any],
        data_dim: int = 5,
        angle: float = math.pi / 3,
    ) -> None:
        self.num_timesteps = int(data.shape[1])
        self.num_layers = runspecs["constants"]["NUM_LAYERS"]
        self.num_zcells = int(
            runspecs["constants"]["NUM_ZCELLS"] / runspecs["constants"]["NUM_LAYERS"]
        )
        self.num_xcells = runspecs["constants"]["NUM_XCELLS"] - 4

        self.data = data.reshape(
            -1,
            self.num_timesteps,
            self.num_layers,
            self.num_zcells,
            self.num_xcells + 2,
            data_dim,
        )

        self.data = self.data[..., :-1, :]
        self.num_members = self.data.shape[0]
        self.single_feature_shape = (
            self.num_members,
            self.num_timesteps,
            self.num_layers,
            self.num_xcells,
        )
        self.runspecs = runspecs
        self.angle = angle