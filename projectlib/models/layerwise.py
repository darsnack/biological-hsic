import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.tree_util as jtu
import flax.linen as nn
from flax.core.scope import DenyList, union_filters
from flax.core.frozen_dict import freeze

from projectlib.utils import grow_to

class LayerwiseModule(nn.Module):
    def init(self, rngs, *args,
             method = None,
             mutable = DenyList(deny=["intermediates", "layer_acts"]),
             capture_intermediates = False,
             **kwargs):
        return super().init(rngs, *args,
                            method=method,
                            mutable=mutable,
                            capture_intermediates=capture_intermediates,
                            **kwargs)

    def lapply(self, variables, *args,
               rngs = None,
               method = None,
               mutable = False,
               capture_intermediates = False,
               **kwargs):
        y, acts = self.apply(variables, *args,
                             rngs=rngs,
                             method=method,
                             mutable=union_filters(["layer_acts"], mutable),
                             capture_intermediates=capture_intermediates,
                             **kwargs)
        acts = {k: v[0] for k, v in acts["layer_acts"].items()}

        return y, acts

def lvalue_and_grad(fun, argnums = 0, has_aux = False, reduce_axes = ()):
    def _lgrad(*primals):
        ys, vjp_fun, *aux = jax.vjp(fun, *primals,
                                    has_aux=has_aux, reduce_axes=reduce_axes)
        nlayers = len(ys)
        dy = {layer: jnp.squeeze(jnp.eye(nlayers, 1, -i))
              for i, layer in enumerate(ys.keys())}
        dparams = jax.vmap(vjp_fun)(dy)[argnums]
        def _mask_and_reduce(param, mask):
            return jnp.sum(param * grow_to(mask, param.ndim), axis=0)
        grads = {layer: jtu.tree_map(lambda p: _mask_and_reduce(p, dy[output]),
                                     dparams["params"][layer])
                 for layer, output in zip(dparams["params"].keys(), dy.keys())}
        grads = {"params": grads}
        y = jnp.stack(list(ys.values()))

        if has_aux:
            return (y, *aux), grads
        else:
            return y, grads

    return _lgrad
