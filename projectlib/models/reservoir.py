import jax
import jax.numpy as jnp
import jax.random as jrng
import flax.linen as nn
from flax.linen.initializers import Initializer
from typing import Callable, Optional, Any, Tuple

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = jax.Array
InitFn = Callable[[PRNGKey, Shape, Dtype], Array]

def reservoir_uniform_init():
    init_fn = nn.initializers.uniform(2.)
    def init(key, shape, dtype) -> Initializer:
        return init_fn(key, shape, dtype) - 1

    return init

def reservoir_sparse_init(sparsity):
    def init(key, shape, dtype) -> Initializer:
        normal_key, sparse_key = jrng.split(key)
        std = jnp.sqrt(1 / (sparsity * shape[0]))
        x = jrng.normal(normal_key, shape, dtype) * std
        p = jrng.bernoulli(sparse_key, 1 - sparsity, shape)

        return x * p

    return init

class ReservoirCell(nn.RNNCellBase):
    output_size: int
    kernel_init: InitFn = reservoir_uniform_init()
    recurrent_kernel_init: InitFn = reservoir_sparse_init(0.1)
    readout_kernel_init: InitFn = nn.initializers.zeros_init()
    time_constant: float = 10e-3
    time_step: float = 1e-3
    recurrent_strength: float = 1.2
    hidden_noise: float = 5e-2
    output_noise: float = 5e-1
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    rng_collection: str = "reservoir"

    @nn.compact
    def __call__(self, carry, inputs):
        u = carry
        x = inputs
        hidden_features = u.shape[-1]

        dense_r = nn.Dense(features=hidden_features,
                           use_bias=False,
                           dtype=self.dtype,
                           param_dtype=self.param_dtype,
                           kernel_init=self.recurrent_kernel_init,
                           name="r")
        dense_f = nn.Dense(features=hidden_features,
                           use_bias=False,
                           dtype=self.dtype,
                           param_dtype=self.param_dtype,
                           kernel_init=self.kernel_init,
                           name="f")
        dense_i = nn.Dense(features=hidden_features,
                           use_bias=False,
                           dtype=self.dtype,
                           param_dtype=self.param_dtype,
                           kernel_init=self.kernel_init,
                           name="i")
        dense_o = nn.Dense(features=self.output_size,
                           use_bias=False,
                           dtype=self.dtype,
                           param_dtype=self.param_dtype,
                           kernel_init=self.readout_kernel_init,
                           name="o")

        rng = self.make_rng(self.rng_collection)
        hidden_rng, output_rng = jrng.split(rng)

        # compute hidden neuron firing rate
        r = nn.tanh(u)
        xi_hidden = jrng.uniform(hidden_rng, r.shape,
                                 minval=-self.hidden_noise,
                                 maxval=self.hidden_noise)
        r = r + xi_hidden
        # store the firing rate for use later
        self.sow("intermediates", "rstore", r)

        # compute output neuron firing rate
        z = dense_o(r)
        xi_output = jrng.uniform(output_rng, z.shape,
                                 minval=-self.output_noise,
                                 maxval=self.output_noise)
        z = z + xi_output
        # jax.debug.callback(print, (z[0, :4], xi_output[0, :4]), ordered=True)

        # update hidden neuron state
        du = self.recurrent_strength * dense_r(r) + dense_i(x) + dense_f(z)
        u = u + self.time_step * (du - u) / self.time_constant

        return u, z

    @staticmethod
    def initialize_carry(rng, batch_dims, size,
                         init_fn = nn.initializers.zeros_init()):

        return init_fn(rng, (*batch_dims, size))

class Reservoir(nn.Module):
    hidden_size: int
    output_size: int
    time_constant: float = 10e-3
    time_step: float = 1e-3
    recurrent_strength: float = 1.2
    hidden_noise: float = 5e-2
    output_noise: float = 5e-1

    def setup(self):
        self.cell = ReservoirCell(self.output_size,
                                  time_constant=self.time_constant,
                                  time_step=self.time_step,
                                  recurrent_strength=self.recurrent_strength,
                                  hidden_noise=self.hidden_noise,
                                  output_noise=self.output_noise,
                                  name="cell")

    @staticmethod
    def initialize_carry(rng, batch_dims, size):
        return ReservoirCell.initialize_carry(rng, batch_dims, size)

    def __call__(self, inputs):
        *batch_size, _, _ = inputs.shape

        def scan_fn(cell, carry, inputs):
            return cell(carry, inputs)

        carry = Reservoir.initialize_carry(None, batch_size, self.hidden_size)
        _, outputs = nn.scan(scan_fn,
                             variable_axes={"intermediates": 1},
                             variable_broadcast="params",
                             split_rngs={"params": False, "reservoir": False},
                             in_axes=1,
                             out_axes=1)(self.cell, carry, inputs)

        return outputs
