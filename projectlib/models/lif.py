import jax
import jax.numpy as jnp
import jax.random as jrng
import flax.linen as nn
from typing import Optional, Any, Sequence

from projectlib.models.layerwise import LayerwiseModule
from projectlib.models.reservoir import reservoir_uniform_init
from projectlib.utils import flatten

Dtype = Any  # this could be a real type?

class LIFDenseCell(nn.RNNCellBase):
    features: int
    time_constant: float = 10e-3
    time_step: float = 1e-3
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, carry, inputs):
        u = carry
        x = inputs

        dense_i = nn.Dense(features=self.features,
                           use_bias=True,
                           dtype=self.dtype,
                           kernel_init=reservoir_uniform_init(),
                           param_dtype=self.param_dtype,
                           name="i")

        # update hidden neuron state
        u = u + self.time_step * (dense_i(x) - u) / self.time_constant

        # compute hidden neuron firing rate
        r = u

        return u, r

    @staticmethod
    def initialize_carry(rng, batch_dims, size,
                         init_fn = nn.initializers.zeros_init()):

        return init_fn(rng, (*batch_dims, size))

class LIFMLPCell(LayerwiseModule):
    features: Sequence[int]
    nclasses: int
    time_constant: float = 10e-3
    time_step: float = 1e-3
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, us, x):
        x = flatten(x)
        _us = []
        for i, feature in enumerate(self.features):
            u, x = LIFDenseCell(features=feature,
                                time_constant=self.time_constant,
                                time_step=self.time_step)(us[i], x)
            _us.append(u)
            x = nn.tanh(x)
            x = nn.LayerNorm(use_scale=False, use_bias=False)(x)
            self.sow("layer_acts", f"dense_{i}", x)
        u, x = LIFDenseCell(features=self.nclasses,
                            time_constant=self.time_constant,
                            time_step=self.time_step)(us[-1], x)
        _us.append(u)
        self.sow("layer_acts", f"dense_{len(self.features)}", x)

        return _us, x

    @staticmethod
    def initialize_carry(rng, batch_dims, sizes,
                         init_fn = nn.initializers.zeros_init()):
        rngs = jrng.split(rng, len(sizes))

        return [LIFDenseCell.initialize_carry(rng, batch_dims, size, init_fn)
                for rng, size in zip(rngs, sizes)]
