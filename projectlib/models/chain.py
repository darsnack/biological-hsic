import flax.linen as nn
from flax import struct
from typing import Sequence
from functools import partial

from projectlib.models.simple import MLP, CNN
from projectlib.utils import flatten

@struct.dataclass
class Chain:
    models: Sequence[nn.Module]
    flatten: bool = False

    @classmethod
    def from_model(cls, model: nn.Module):
        if isinstance(model, MLP):
            models = [nn.Sequential([nn.Dense(features=f, dtype=model.dtype),
                                     nn.relu])
                      for f in model.features]
            models.append(nn.Dense(features=model.nclasses, dtype=model.dtype))

            return cls(models, True)
        elif isinstance(model, CNN):
            pool_size = (model.pooling_factor, model.pooling_factor)
            pool_fn = partial(nn.avg_pool, window_shape=pool_size, strides=pool_size)
            models = [nn.Sequential([nn.Conv(features=f,
                                             kernel_size=(3, 3),
                                             padding=1,
                                             dtype=model.dtype),
                                     nn.relu,
                                     pool_fn])
                      for f in model.features]
            models.append(nn.Sequential([flatten, nn.Dense(features=model.nclasses)]))

            return cls(models, False)
        else:
            raise ValueError(f"Cannot separate {type(model)} into Chain")

    def init(self, rngs, *args, **kwargs):
        params = []
        if self.flatten:
            args = tuple(flatten(a) for a in args)
        for model in self.models:
            *args, ps = model.init_with_output(rngs, *args, **kwargs)
            params.append(ps)

        return params

    def get_apply_fns(self):
        return [model.apply for model in self.models]
