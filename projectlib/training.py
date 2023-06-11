import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.tree_util as jtu
from clu import metrics
from flax import struct
from flax.training import train_state
from flax.traverse_util import path_aware_map
from flax.core.frozen_dict import freeze
from typing import Dict, Any, Tuple
from dataclasses import dataclass

from projectlib.utils import maybe
from projectlib.logging import PrintLogger
from projectlib.hsic import kernel_matrix, global_error

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
        apply_fn = maybe(apply_fn, model.apply)

        return cls.create(apply_fn=apply_fn,
                          params=params,
                          model_state=model_state,
                          tx=opt,
                          rngs=rngs,
                          metrics=metrics)

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
    def rmhebb_step(state: TrainState, batch, rng = None):
        # compute global error signal
        xs, ys, zs = batch
        sigmax, sigmay, sigmaz = sigmas
        kx = kernel_matrix(xs, sigmax)
        ky = kernel_matrix(ys, sigmay)
        kz = kernel_matrix(zs, sigmaz)
        xi = gain * global_error(kx, ky, kz, zs, gamma, sigmaz)
        xi = jnp.expand_dims(xi, axis=0)
        # compute input signal
        inputs = jnp.concatenate([xs[0], ys[0], zs[0]], axis=-1)
        inputs = jnp.expand_dims(inputs, axis=0)

        # define each time step
        def step(_, carry):
            train_state, lpf_state = carry
            # run reservoir forward
            (model_state, outputs), aux = train_state.apply_fn(
                train_state.params, train_state.model_state, inputs,
                rngs=train_state.rngs,
                method="step",
                mutable="intermediates"
            )
            rs = aux["intermediates"]["cell"]["rstore"][0]
            train_state = train_state.replace(model_state=model_state)
            # update LPF value
            lpf_errors, lpf_outputs = lpf_state
            mse = -jnp.sum((outputs - xi) ** 2)
            lpf_errors = lpf(lpf_errors, mse)
            lpf_outputs = lpf(lpf_outputs, outputs)
            # compute and apply update
            reward = jax.lax.convert_element_type(mse > lpf_errors, jnp.int_)
            dW = -reward * (outputs - lpf_outputs) * jnp.transpose(rs)
            dWkey = ("params", "cell", "o", "kernel")
            grads = freeze(path_aware_map(
                lambda k, v: dW if k == dWkey else jnp.zeros_like(v),
                train_state.params
            ))
            train_state = train_state.apply_gradients(grads=grads)

            return train_state, (lpf_errors, lpf_outputs)

        # run over batch for ntimesteps
        lpf_errors = jnp.zeros_like(xi, shape=())
        lpf_outputs = jnp.zeros_like(xi)
        init_carry = (state, (lpf_errors, lpf_outputs))
        state, (loss, _) = jax.lax.fori_loop(0, ntimesteps, step, init_carry)
        # for i in range(ntimesteps):
        #     carry = step(i, carry)
        # state, (loss, _) = carry

        return loss, state

    return rmhebb_step

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
                test_state = metrics_fn(test_state, batch, rng_metric)

            # average metrics
            for metric, value in state.metrics.compute().items():
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
