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

    def compute(self):
        if len(self.loss.count.devices()) > 1:
            return super(Metrics, self.unreplicate()).compute()
        elif self.loss.count.ndim > 0:
            return jax.vmap(lambda m: super(Metrics, m).compute())(self)
        else:
            return super(Metrics, self).compute()

    def init_history(self):
        return {
            "train": {k: [] for k in self.__annotations__.keys()},
            "test": {k: [] for k in self.__annotations__.keys()},
        }

class TrainState(train_state.TrainState):
    metrics: Metrics
    rngs: Dict[str, Array]
    model_state: Any
    aux_state: Any

    @classmethod
    def from_model(cls, model, dummy_input, opt, rngs,
                   param_init = None,
                   model_state = None,
                   aux_state = None,
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
                          aux_state=aux_state,
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

def create_rmhebb_step(ntimesteps, gamma, sigma):
    @jax.jit
    def rmhebb_step(state: TrainState, batch, _ = None):
        # compute global error signal
        xs, ys, zs = batch
        # compute input signal
        inputs = jnp.expand_dims(jnp.concatenate([xs[-1], ys[-1], zs[-1]], axis=0), axis=0)
        # compute target signal
        kx = kernel_matrix(xs, sigma)
        ky = kernel_matrix(ys, sigma)
        kz = kernel_matrix(zs, sigma)
        targets = global_error(kx, ky, kz, zs, gamma, sigma)

        # define each time step
        def step(_, carry):
            losses, train_state = carry
            # run reservoir forward
            train_state = train_state.split_rngs()
            (model_state, outputs), aux = train_state.apply_fn(
                train_state.params, train_state.model_state, inputs,
                rngs=train_state.rngs,
                mutable="intermediates"
            )
            rs = aux["intermediates"]["rstore"][0]
            train_state = train_state.replace(model_state=model_state)
            # compute and apply update
            dW = (outputs - targets) * jnp.transpose(rs)
            # jax.debug.callback(print, reward, ordered=True)
            dWkey = ("params", "o", "kernel")
            grads = path_aware_map(
                lambda k, v: dW if k == dWkey else jnp.zeros_like(v),
                train_state.params
            )
            train_state = train_state.apply_gradients(grads=grads)
            # compute loss
            losses += jnp.mean((outputs - targets) ** 2)

            return losses, train_state

        # run over batch for ntimesteps
        losses = jnp.zeros_like(targets, shape=())
        init_carry = (losses, state)
        losses, state = jax.lax.fori_loop(0, ntimesteps, step, init_carry)
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

def create_hsic_step(loss_fn, gamma):
    @jax.jit
    def train_step(state: TrainState, batch, _ = None):
        xs, ys = batch
        sigmas = state.aux_state
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
        # state = state.replace(aux_state=(0.9 * sigmas[0], 0.9 * sigmas[1], sigmas[2]))

        return losses[-1], state

    return train_step

def create_biohsic_step(loss_fn, gamma, sigmas):
    @jax.jit
    def train_step(state: TrainState, batch, _ = None):
        xs, ys = batch
        bs = ys.shape[0]
        # compute input and output kernel matrices ahead of time
        sigmax, sigmay, sigmaz = sigmas
        kx = kernel_matrix(flatten(xs), sigmax)
        ky = kernel_matrix(flatten(ys), sigmay)
        # apply model forward and compute layerwise vjp along the way
        def _fwd(params):
            _, acts = state.apply_fn(params, xs, rngs=state.rngs)

            return acts
        zs, vjp_fun = jax.vjp(_fwd, state.params)
        # pushback modulated perturbations through the vjp
        def _build_dy(i, z):
            kz = kernel_matrix(flatten(z), sigmaz)
            xi = global_error(kx, ky, kz, z, gamma, sigmaz)
            if i == len(zs) - 1:
                dz = jax.grad(lambda _z: jnp.mean(loss_fn(_z, ys)))(z)
            else:
                dz = jnp.concatenate([jnp.zeros_like(z, shape=(bs - 1, *z.shape[1:])),
                                      xi * jnp.ones_like(z, shape=(1, *z.shape[1:]))],
                                     axis=0)
            dy = jnp.stack([dz if j == i else jnp.zeros_like(z)
                            for j in range(len(zs))], axis=0)

            return dy
        dys = {layer: _build_dy(i, zs[layer])
               for i, layer in enumerate(zs.keys())}
        grads = jax.vmap(vjp_fun)(dys)[0]
        grads = {layer: jtu.tree_map(lambda p: p[i], grads["params"][layer])
                 for i, layer in enumerate(grads["params"].keys())}
        grads = {"params": grads}
        loss = jnp.mean(loss_fn(list(zs.values())[-1], ys))

        # grad_norms = [[jnp.linalg.norm(jnp.reshape(g, -1))
        #                for g in jtu.tree_leaves(gs)]
        #               for gs in grads["params"].values()]
        # def log(grad_norms):
        #     # wandb.log({"zs": wandb.Histogram(zs)}, commit=False)
        #     wandb.log({f"gradnorm_{i}": {str(j): grad_norm_j
        #                                  for j, grad_norm_j in enumerate(grad_norm)}
        #                for i, grad_norm in enumerate(grad_norms)}, commit=False)
        # jax.debug.callback(log, grad_norms, ordered=True)

        # update model
        state = state.apply_gradients(grads=grads)

        return loss, state

    return train_step

def create_lif_biohsic_step(loss_fn, gamma, sigmas, ntimesteps):
    @jax.jit
    def train_step(state: TrainState, batch, _ = None):
        xs, ys = batch
        bs = ys.shape[0]
        # compute input and output kernel matrices ahead of time
        sigmax, sigmay, sigmaz = sigmas
        kx = kernel_matrix(flatten(xs), sigmax)
        ky = kernel_matrix(flatten(ys), sigmay)

        def step(_, carry):
            loss, state = carry
            # apply model forward and compute layerwise vjp along the way
            def _fwd(params):
                (model_state, _), acts = state.apply_fn(params, state.model_state, xs,
                                         rngs=state.rngs)

                return acts, model_state
            zs, vjp_fun, model_state = jax.vjp(_fwd, state.params, has_aux=True)
            state = state.replace(model_state=model_state)
            # pushback modulated perturbations through the vjp
            def _build_dy(i, z):
                kz = kernel_matrix(flatten(z), sigmaz)
                xi = global_error(kx, ky, kz, z, gamma, sigmaz)
                if i == len(zs) - 1:
                    dz = jax.grad(lambda _z: jnp.mean(loss_fn(_z, ys)))(z)
                else:
                    dz = jnp.concatenate([jnp.zeros_like(z, shape=(bs - 1, *z.shape[1:])),
                                          xi * jnp.ones_like(z, shape=(1, *z.shape[1:]))],
                                         axis=0)
                dy = jnp.stack([dz if j == i else jnp.zeros_like(z)
                                for j in range(len(zs))], axis=0)

                return dy
            # print([z.shape for z in zs.values()])
            dys = {layer: _build_dy(i, zs[layer])
                for i, layer in enumerate(zs.keys())}
            grads = jax.vmap(vjp_fun)(dys)[0]
            grads = {layer: jtu.tree_map(lambda p: p[i], grads["params"][layer])
                    for i, layer in enumerate(grads["params"].keys())}
            grads = {"params": grads}
            loss += jnp.mean(loss_fn(list(zs.values())[-1], ys))

            def log(grad_norms):
                # wandb.log({"zs": wandb.Histogram(zs)}, commit=False)
                wandb.log({f"gradnorm_{i}": {str(j): grad_norm_j
                                            for j, grad_norm_j in enumerate(grad_norm)}
                        for i, grad_norm in enumerate(grad_norms)}, commit=False)
            grad_norms = [[jnp.linalg.norm(jnp.reshape(g, -1))
                           for g in jtu.tree_leaves(gs)]
                          for gs in grads["params"].values()]
            jax.debug.callback(log, grad_norms, ordered=True)

            # update model
            state = state.apply_gradients(grads=grads)

            return loss, state

        loss = jnp.zeros(())
        loss, state = jax.lax.fori_loop(0, ntimesteps, step, (loss, state))
        loss = loss / ntimesteps

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

    # vmap helpers for multiple parallel models
    if rng.ndim > 1:
        def rng_split(key, n = 2):
            splits = jax.vmap(jrng.split, in_axes=(0, None))(key, n)
            return tuple(splits[:, i, :] for i in range(n))

        @jax.vmap
        def split_state_rngs(state):
            return state.split_rngs()
    else:
        def rng_split(key, n = 2):
            return jrng.split(key, n)

        def split_state_rngs(state):
            return state.split_rngs()

    epoch_len = len(data["train"])
    for epoch in range(nepochs):
        # run epoch
        for i, batch in enumerate(data["train"].as_numpy_iterator()):
            batch = batch_values(batch)
            rng, rng_step, rng_metric = rng_split(rng, 3)
            state = split_state_rngs(state)
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
                rng, rng_metric = rng_split(rng)
                test_state = split_state_rngs(test_state)
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
