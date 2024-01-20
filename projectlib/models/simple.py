import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Any

from projectlib.models.layerwise import LayerwiseModule
from projectlib.utils import flatten

Dtype = Any

class MLP(LayerwiseModule):
    """A simple MLP model."""
    features: Sequence[int]
    nclasses: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = flatten(x)
        for i, feature in enumerate(self.features):
            x = nn.Dense(features=feature, dtype=self.dtype)(x)
            x = nn.relu(x)
            x = nn.LayerNorm(use_scale=False, use_bias=False)(x)
            self.sow("layer_acts", f"dense_{i}", x)
        x = nn.Dense(features=self.nclasses)(x)
        self.sow("layer_acts", f"dense_{len(self.features)}", x)

        return x

class CNN(LayerwiseModule):
    """A simple CNN model."""
    features: Sequence[int]
    nclasses: int
    pooling_factor: int = 2
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        for i, feature in enumerate(self.features):
            x = nn.Conv(features=feature,
                        kernel_size=(3, 3),
                        padding=1,
                        dtype=self.dtype)(x)
            x = nn.relu(x)
            x = nn.LayerNorm(use_scale=False, use_bias=False, dtype=self.dtype)(x)
            pool_size = (self.pooling_factor, self.pooling_factor)
            x = nn.avg_pool(x, window_shape=pool_size, strides=pool_size)
            self.sow("layer_acts", f"conv_{i}", x)
            # x = nn.Dropout(rate=0.2, deterministic=not train)(x)
        x = flatten(x)
        x = nn.Dense(features=self.nclasses)(x)
        self.sow("layer_acts", f"dense_{len(self.features)}", x)

        return x
