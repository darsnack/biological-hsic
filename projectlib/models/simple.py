import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Any

from projectlib.utils import flatten

Dtype = Any

class MLP(nn.Module):
    """A simple MLP model."""
    features: Sequence[int]
    nclasses: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = flatten(x)
        for feature in self.features:
            x = nn.Dense(features=feature, dtype=self.dtype)(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.nclasses)(x)

        return x

class CNN(nn.Module):
    """A simple CNN model."""
    features: Sequence[int]
    nclasses: int
    pooling_factor: int = 2
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        for feature in self.features:
            x = nn.Conv(features=feature,
                        kernel_size=(3, 3),
                        padding=1,
                        use_bias=False,
                        dtype=self.dtype)(x)
            # x = nn.LayerNorm(dtype=self.dtype)(x)
            x = nn.relu(x)
            pool_size = (self.pooling_factor, self.pooling_factor)
            x = nn.avg_pool(x, window_shape=pool_size, strides=pool_size)
            # x = nn.Dropout(rate=0.2, deterministic=not train)(x)
        x = flatten(x)
        x = nn.Dense(features=self.nclasses)(x)

        return x
