import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.tree_util as jtu
import hydra
from orbax.checkpoint import SaveArgs, RestoreArgs
from omegaconf import DictConfig

def maybe(this, that):
    return that if this is None else this

def flatten(x):
    return jnp.reshape(x, (x.shape[0], -1))

def grow_dims(x, before, after):
    before_dims = tuple(i for i in range(before))
    after_dims = tuple(i for i in range(x.ndim + before, x.ndim + before + after))

    return jnp.expand_dims(x, (*before_dims, *after_dims))

def grow_to(x, to):
    return grow_dims(x, before=0, after=to - x.ndim)

def setup_rngs(seed, keys = ["model", "train"]):
    root_rng = jrng.PRNGKey(seed) if not isinstance(seed, jax.Array) else seed
    rngs = {k: rng for k, rng in zip(keys, jrng.split(root_rng, len(keys)))}

    return {"root": root_rng, **rngs}

def value_and_vgrad(f):
    def _value_and_vgrad(*args, **kwargs):
        out, pullback = jax.vjp(f, *args, **kwargs)
        grad = pullback(jnp.ones(out.shape))[0]

        return out, grad

    return _value_and_vgrad

def ckpt_save_kwargs(ckpt, **kwargs):
    return {
        "save_args": jtu.tree_map(lambda _: SaveArgs(**kwargs), ckpt)
    }

def ckpt_restore_kwargs(ckpt, **kwargs):
    return {
        "restore_args": jtu.tree_map(lambda _: RestoreArgs(**kwargs), ckpt)
    }

def instantiate_schedule(cfg: DictConfig, steps_per_epoch):
    if (cfg._target_ == "optax.exponential_decay") and cfg.staircase:
        transition_steps = steps_per_epoch * cfg.transition_steps
        return hydra.utils.instantiate(cfg, transition_steps=transition_steps)
    else:
        return hydra.utils.instantiate(cfg)

def instantiate_optimizer(cfg: DictConfig, steps_per_epoch):
    if isinstance(cfg.learning_rate, DictConfig):
        lr = instantiate_schedule(cfg.learning_rate, steps_per_epoch)

        return hydra.utils.instantiate(cfg, learning_rate=lr)
    else:
        return hydra.utils.instantiate(cfg)
