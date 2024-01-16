import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.tree_util as jtu
import wandb
from clu import metrics
from flax import struct
from flax.training import train_state
from flax.traverse_util import path_aware_map
from flax.core.frozen_dict import freeze
from typing import Dict, Any
from dataclasses import dataclass
from functools import partial

from projectlib.utils import maybe, flatten
from projectlib.logging import PrintLogger
from projectlib.hsic import kernel_matrix, global_error, hsic_bottleneck
from projectlib.models.layerwise import lvalue_and_grad

Array = Any

def batch_values(batch):
    if isinstance(batch, Dict):
        return tuple(batch.values())
    else:
        return batch

@struct.dataclass
class TraceMetric(metrics.Metric):
    values: Array

    def merge(self, other):
        values = jnp.concatenate([self.values, other.values])

        return type(self)(values)

    def reduce(self):
        return self.values[1:]

    def compute(self):
        return self.values[1:]

    @classmethod
    def from_output(cls, name, shape, dtype = jnp.float32):
        @struct.dataclass
        class FromOutputs(cls):
            @classmethod
            def empty(cls):
                return cls(values=jnp.zeros(shape=(1, *shape), dtype=dtype))

            @classmethod
            def from_model_output(cls, **model_output):
                value = jnp.expand_dims(jnp.array(model_output[name]), axis=0)

                return cls(value)

        return FromOutputs

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Average.from_output('accuracy')
    loss: metrics.Average.from_output('loss')

    @classmethod
    def with_aux(cls, **metrics):
        return struct.dataclass(type(
            "_InlineMetrics",
            (Metrics,),
            {"__annotations__": {**cls.__annotations__, **metrics}})
        )

    @classmethod
    def with_hsic(cls, nlayers):
        return Metrics.with_aux(
            **{f"hsic_layer{i}": metrics.Average.from_output(f"hsic{i}")
               for i in range(nlayers)},
            **{f"hsicx_layer{i}": metrics.Average.from_output(f"hsicx{i}")
               for i in range(nlayers)},
            **{f"hsicy_layer{i}": metrics.Average.from_output(f"hsicy{i}")
               for i in range(nlayers)}
        )

    def init_history(self):
        return {
            "train": {k: [] for k in self.__annotations__.keys()},
            "test": {k: [] for k in self.__annotations__.keys()},
        }

class TrainState(train_state.TrainState):
    metrics: Metrics
    rngs: Dict[str, Array]
    model_state: Any

    @classmethod
    def from_model(cls, model, dummy_input, opt, rngs,
                   param_init = None,
                   model_state = None,
                   metrics = None,
                   apply_fn = None):
        _init = maybe(param_init, model.init)
        if isinstance(dummy_input, tuple):
            params = _init(rngs, *dummy_input)
        else:
            params = _init(rngs, dummy_input)
        metrics = Metrics.empty() if metrics is None else metrics
        apply_fn = model.apply if apply_fn is None else apply_fn

        return cls.create(apply_fn=apply_fn,
                          params=params,
                          model_state=model_state,
                          tx=opt,
                          rngs=rngs,
                          metrics=metrics)

    def split_rngs(self):
        rngs = {k: jrng.split(v, 1)[0] for k, v in self.rngs.items()}

        return self.replace(rngs=rngs)

    def reset(self, tx = None):
        _tx = self.tx if tx is None else tx
        opt_state = _tx.init(self.params)

        return self.replace(step=0, tx=_tx, opt_state=opt_state)

@dataclass
class LowPassFilter:
    time_constant: float
    time_step: float = 1e-3

    def __call__(self, xavg, x):
        tau = self.time_step / self.time_constant

        return (1 - tau) * xavg + tau * x

def create_rmhebb_step(ntimesteps, gamma, sigmas, gain, lpf: LowPassFilter):
    @jax.jit
    def rmhebb_step(state: TrainState, batch, _ = None):
        # compute global error signal
        xs, ys, zs = batch
        sigmax, sigmay, sigmaz = sigmas
        kx = kernel_matrix(xs, sigmax)
        ky = kernel_matrix(ys, sigmay)
        kz = kernel_matrix(zs, sigmaz)
        xi = gain * global_error(kx, ky, kz, zs, gamma, sigmaz)
        xi = jnp.expand_dims(xi, axis=0)
        # compute input signal
        inputs = jnp.concatenate([xs[-1], ys[-1], zs[-1]], axis=-1)
        inputs = jnp.expand_dims(inputs, axis=0)

        # define each time step
        def step(_, carry):
            losses, train_state, lpf_state = carry
            # run reservoir forward
            train_state = train_state.split_rngs()
            (model_state, outputs), aux = train_state.apply_fn(
                train_state.params, train_state.model_state, inputs,
                rngs=train_state.rngs,
                mutable="intermediates"
            )
            rs = aux["intermediates"]["rstore"][0]
            train_state = train_state.replace(model_state=model_state)
            # update LPF value
            lpf_errors, lpf_outputs = lpf_state
            errors = -jnp.sum((outputs - xi) ** 2)
            lpf_errors = lpf(lpf_errors, errors)
            lpf_outputs = lpf(lpf_outputs, outputs)
            # compute and apply update
            # reward = jax.lax.convert_element_type(errors > lpf_errors, jnp.int_)
            reward = errors > lpf_errors
            dW = reward * (lpf_outputs - outputs) * jnp.transpose(rs)
            dWkey = ("params", "o", "kernel")
            grads = freeze(path_aware_map(
                lambda k, v: dW if k == dWkey else jnp.zeros_like(v),
                train_state.params
            ))
            train_state = train_state.apply_gradients(grads=grads)
            # compute loss
            losses += -errors / jnp.size(xi)

            return losses, train_state, (lpf_errors, lpf_outputs)

        # run over batch for ntimesteps
        losses = jnp.zeros_like(xi, shape=())
        lpf_errors = jnp.zeros_like(xi, shape=())
        lpf_outputs = jnp.zeros_like(xi)
        init_carry = (losses, state, (lpf_errors, lpf_outputs))
        losses, state, _ = jax.lax.fori_loop(0, ntimesteps, step, init_carry)
        # carry = init_carry
        # for i in range(ntimesteps):
        #     carry = step(i, carry)
        # losses, state, _ = carry

        return losses / ntimesteps, state

    return rmhebb_step

def grad_norm(gs):
    leaves = jtu.tree_leaves(gs)

    return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))

def clip_grad_norm(gs, max_norm):
    norm = grad_norm(gs)
    normalize = lambda g: jnp.where(norm < max_norm, g, g * (max_norm / norm))

    return jtu.tree_map(normalize, gs)

def create_hsic_step(loss_fn, gamma, sigmas):
    @jax.jit
    def train_step(state: TrainState, batch, _ = None):
        xs, ys = batch
        # apply model forward and compute layerwise gradients along the way
        def compute_loss(params):
            _, acts = state.apply_fn(params, xs, rngs=state.rngs)
            hsic_losses = {
                k: hsic_bottleneck(xs, ys, zs, gamma, *sigmas)[0] if i < len(acts) - 1
                   else jnp.mean(loss_fn(zs, ys))
                for i, (k, zs) in enumerate(acts.items())
            }

            return hsic_losses
        losses, grads = lvalue_and_grad(compute_loss)(state.params)

        grad_norms = [[jnp.linalg.norm(jnp.reshape(g, -1))
                       for g in jtu.tree_leaves(gs)]
                      for gs in grads["params"].values()]
        def log(grad_norms):
            # wandb.log({"zs": wandb.Histogram(zs)}, commit=False)
            wandb.log({f"gradnorm_{i}": {str(j): grad_norm_j
                                         for j, grad_norm_j in enumerate(grad_norm)}
                       for i, grad_norm in enumerate(grad_norms)}, commit=False)
        jax.debug.callback(log, grad_norms, ordered=True)

        # update model
        # grads = jtu.tree_map(lambda g: jnp.clip(g, -1, 1), grads)
        state = state.apply_gradients(grads=grads)

        return losses[-1], state

    return train_step

def create_biohsic_step(loss_fn, gamma, sigmas, flatten_input = False):
    @jax.jit
    def train_step(state: TrainState, batch, _ = None):
        xs, ys = batch
        bs = ys.shape[0]
        # compute input and output kernel matrices ahead of time
        sigmax, sigmay, sigmaz = sigmas
        kx = kernel_matrix(flatten(xs), sigmax)
        ky = kernel_matrix(flatten(ys), sigmay)
        # apply model forward and compute layerwise gradients along the way
        zs = []
        updates = []
        zs_last = flatten(xs) if flatten_input else xs
        for apply_fn, params in zip(state.apply_fn[:-1], state.params[:-1]):
            # run forward step
            apply_fn = partial(apply_fn, rngs=state.rngs)
            zs_last, pullback = jax.vjp(apply_fn, params, zs_last)
            # compute global error
            kz = kernel_matrix(flatten(zs_last), sigmaz)
            xi = global_error(kx, ky, kz, zs_last, gamma, sigmaz)
            zs_shape = zs_last.shape[1:]
            # compute gradient accounting for global error
            zpertub = jnp.concatenate([jnp.zeros_like(zs_last, shape=(bs - 1, *zs_shape)),
                                       xi * jnp.ones_like(zs_last, shape=(1, *zs_shape))],
                                       axis=0)
            updates_last = pullback(zpertub)[0]
            zs.append(zs_last)
            # updates.append(jtu.tree_map(lambda x: x * 1e-3, updates_last))
            updates.append(updates_last)
        # apply the final layer using BP
        def compute_loss(params):
            yhats = state.apply_fn[-1](params, zs_last, rngs=state.rngs)

            return jnp.mean(loss_fn(yhats, ys))
        grad_fn = jax.value_and_grad(compute_loss)
        loss, out_grads = grad_fn(state.params[-1])

        # grad_norms = [unfreeze(jtu.tree_map(lambda x: jnp.linalg.norm(jnp.reshape(x, -1)), g))
        #               for g in state.params]
        # # grad_norms.append(unfreeze(jtu.tree_map(lambda x: jnp.linalg.norm(jnp.reshape(x, -1)),
        # #                                out_grads)))
        # print(grad_norms)
        # # wandb.log({"grads": wandb.Histogram(grad_norms[0])}, commit=False)

        # update model
        state = state.apply_gradients(grads=[*updates, out_grads])

        return loss, state

    return train_step

def create_train_step(loss_fn, batch_stats = False):
    # if batch_stats are calculated, then we need to augment the apply_fn
    if batch_stats:
        @jax.jit
        def train_step(state: TrainState, batch, _ = None):
            *xs, ys = batch
            def compute_loss(params):
                yhats, aux = state.apply_fn(params, *xs,
                                            rngs=state.rngs,
                                            train=True,
                                            mutable=["batch_stats"])

                return jnp.mean(loss_fn(yhats, ys)), aux

            grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
            (loss, aux), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(params=state.params.copy(aux))

            return loss, state
    else:
        @jax.jit
        def train_step(state: TrainState, batch, _ = None):
            *xs, ys = batch
            def compute_loss(params):
                yhats = state.apply_fn(params, *xs, rngs=state.rngs)

                return jnp.mean(loss_fn(yhats, ys))

            grad_fn = jax.value_and_grad(compute_loss)
            loss, grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)

            return loss, state

    return train_step

def fit(data, state: TrainState, step_fn, metrics_fn,
        rng = None,
        save_fn = None,
        nepochs = 1,
        epoch_logging = True,
        step_log_interval = 100,
        logger = PrintLogger()):
    metric_history = state.metrics.init_history()
    rng = maybe(rng, jrng.PRNGKey(0))

    epoch_len = len(data["train"])
    for epoch in range(nepochs):
        # run epoch
        for i, batch in enumerate(data["train"].as_numpy_iterator()):
            batch = batch_values(batch)
            rng, rng_step, rng_metric = jrng.split(rng, 3)
            state = state.split_rngs()
            loss, state = step_fn(state, batch, rng_step)
            state = metrics_fn(state, batch, rng_metric)
            if (step_log_interval is not None) and (i % step_log_interval == 0):
                logger.log({"epoch": epoch, "step": i, "loss": loss},
                           commit=(i < epoch_len - 1))

        # average metrics
        for metric, value in state.metrics.compute().items():
            metric_history["train"][metric].append(value)
        state = state.replace(metrics=state.metrics.empty())

        # run test validation
        if "test" in data.keys():
            test_state = state
            for i, batch in enumerate(data["test"].as_numpy_iterator()):
                batch = batch_values(batch)
                rng, rng_metric = jrng.split(rng)
                test_state = test_state.split_rngs()
                test_state = metrics_fn(test_state, batch, rng_metric)

            # average metrics
            for metric, value in test_state.metrics.compute().items():
                metric_history["test"][metric].append(value)

        # run save function
        ckpt = {"train_state": state, "metrics_history": metric_history}
        save_fn(epoch, ckpt)

        # log outputs
        if epoch_logging == "summary":
            train_logs = {k: jnp.mean(v[-1])
                          for k, v in metric_history["train"].items()}
            if "test" in data.keys():
                test_logs = {k: jnp.mean(v[-1])
                             for k, v in metric_history["test"].items()}
            else:
                test_logs = {}
        elif epoch_logging:
            train_logs = {k: v[-1] for k, v in metric_history["train"].items()}
            if "test" in data.keys():
                test_logs = {k: v[-1] for k, v in metric_history["test"].items()}
            else:
                test_logs = {}
        else:
            train_logs = {}
            test_logs = {}
        logger.log({"epoch": epoch,
                    "step": epoch_len - 1,
                    "train metrics": train_logs,
                    "test metrics": test_logs})

    return state, metric_history
