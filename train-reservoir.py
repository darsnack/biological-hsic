import os

# this stops jax from stealing all the memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.tree_util as jtu
import tensorflow as tf
import optax
import seaborn as sns
import hydra
from omegaconf import DictConfig
from math import sqrt
# from clu.metrics import CollectingMetric
from functools import partial
from orbax.checkpoint import (CheckpointManager,
                              CheckpointManagerOptions,
                              PyTreeCheckpointer,
                              SaveArgs)

from projectlib.utils import setup_rngs, instantiate_schedule
from projectlib.data import build_dataloader
from projectlib.models.reservoir import Reservoir
from projectlib.hsic import kernel_matrix, global_error
from projectlib.training import (TrainState,
                                 Metrics,
                                 TraceMetric,
                                 fit,
                                 create_rmhebb_step,
                                 LowPassFilter)

@hydra.main(config_path="./configs", config_name="train-reservoir", version_base=None)
def main(cfg: DictConfig):
    # setup rngs
    rngs = setup_rngs(cfg.seed, keys=["model", "train", "data"])

    # setup dataloaders
    xdata_key, ydata_key, zdata_key = jrng.split(rngs["data"], 3)
    xdata = jrng.uniform(xdata_key, (cfg.data.nsamples, cfg.data.xdim))
    ydata = jrng.uniform(ydata_key, (cfg.data.nsamples, cfg.data.ydim))
    zdata = jrng.uniform(zdata_key, (cfg.data.nsamples, cfg.data.zdim))
    loader = build_dataloader({"x": xdata, "y": ydata, "z": zdata},
                              batch_size=cfg.data.batchsize,
                              window_shift=1)

    # setup model
    model = Reservoir(cfg.model.nhidden, cfg.data.zdim,
                      time_constant=cfg.model.time_constant,
                      time_step=cfg.training.time_step,
                      recurrent_strength=cfg.model.recurrent_strength,
                      hidden_noise=cfg.model.hidden_noise,
                      output_noise=cfg.model.output_noise)

    # initialize randomness
    tf.random.set_seed(cfg.seed) # deterministic data iteration
    param_key, reservoir_key, state_key = jrng.split(rngs["model"], 3)
    init_keys = {"params": param_key,
                 "reservoir": reservoir_key}

    # create optimizer
    schedule = instantiate_schedule(cfg.schedule, len(loader))
    opt = optax.sgd(learning_rate=schedule)
    # create training state (initialize parameters)
    dummy_input = jnp.ones((1, cfg.data.ntimesteps,
                            cfg.data.xdim + cfg.data.ydim + cfg.data.zdim))
    state_init = Reservoir.initialize_carry(state_key,
                                            batch_dims=(1,),
                                            size=cfg.model.nhidden)
    _Metrics = Metrics.with_aux(
        target=TraceMetric.from_output("target", (1, cfg.data.zdim)),
        output=TraceMetric.from_output("output", (1, cfg.data.zdim))
    )
    train_state = TrainState.from_model(model, dummy_input, opt, init_keys,
                                        model_state=state_init,
                                        metrics=_Metrics.empty())
    # create training step
    sigmas = tuple(0.25 * sqrt(d)
                   for d in (cfg.data.xdim, cfg.data.ydim, cfg.data.zdim))
    lpf = LowPassFilter(cfg.training.lpf_time_constant, cfg.training.time_step)
    gain = 1
    train_step = create_rmhebb_step(ntimesteps=cfg.data.ntimesteps,
                                    gamma=cfg.hsic.gamma,
                                    sigmas=sigmas,
                                    gain=gain,
                                    lpf=lpf)
    # create evaluation step
    @jax.jit
    def metric_step(state: TrainState, batch, rng):
        # compute global error signal
        xs, ys, zs = batch
        sigmax, sigmay, sigmaz = sigmas
        kx = kernel_matrix(xs, sigmax)
        ky = kernel_matrix(ys, sigmay)
        kz = kernel_matrix(zs, sigmaz)
        xi = gain * global_error(kx, ky, kz, zs, cfg.hsic.gamma, sigmaz)
        xi = jnp.expand_dims(xi, axis=0)
        # compute input signal
        inputs = jnp.concatenate([xs[0], ys[0], zs[0]], axis=-1)
        inputs = jnp.expand_dims(inputs, axis=0)

        # define each time step
        def step(_, carry):
            train_state, mse, lpf_outputs = carry
            # run reservoir forward
            model_state, outputs = train_state.apply_fn(
                train_state.params, train_state.model_state, inputs,
                rngs=train_state.rngs,
                method="step"
            )
            train_state = train_state.replace(model_state=model_state)
            # compute mse
            lpf_outputs = lpf(lpf_outputs, outputs)
            mse = mse + jnp.mean((lpf_outputs - xi) ** 2)

            return train_state, mse, lpf_outputs

        # run over batch for ntimesteps
        init_carry = (state, jnp.zeros_like(xi, shape=()), jnp.zeros_like(xi))
        state, loss, outputs = jax.lax.fori_loop(0,
                                                 cfg.data.ntimesteps,
                                                 step,
                                                 init_carry)
        loss = loss / cfg.data.ntimesteps
        metrics_updates = state.metrics.single_from_model_output(
            loss = loss, target = xi, output = outputs, accuracy = 0
        )
        metrics = state.metrics.merge(metrics_updates)
        state = state.replace(metrics=metrics)

        return state

    # def metric_step(state: TrainState, batch, rng):
    #     print("hello")
    #     state, metrics_updates = metric_step_jit(state, batch, rng)
    #     # metrics = state.metrics.merge(metrics_updates)
    #     # state = state.replace(metrics=metrics)

    #     return state

    # create checkpointing utility
    ckpt_opts = CheckpointManagerOptions(
        create=True,
        save_interval_steps=cfg.checkpointing.rate,
        max_to_keep=3
    )
    ckpt_mgr = CheckpointManager(os.sep.join([os.getcwd(),
                                              cfg.checkpointing.path]),
                                 PyTreeCheckpointer(),
                                 ckpt_opts)
    # save_kwargs = {
    #     "save_args": {
    #         "train_state": jtu.tree_map(lambda _: SaveArgs(aggregate=True),
    #                                     train_state),
    #         "metrics_history": jtu.tree_map(lambda _: SaveArgs(aggregate=True),
    #                                         train_state.metrics.init_history())
    #     }
    # }
    # save_fn = partial(ckpt_mgr.save, save_kwargs=save_kwargs)

    # create logger
    logger = hydra.utils.instantiate(cfg.logger)
    logger.log_config(cfg)

    # run training
    final_state, trace = fit({"train": loader},
                             train_state,
                             train_step,
                             metric_step,
                             save_fn=ckpt_mgr.save,
                             rng=rngs["train"],
                             nepochs=cfg.training.nepochs,
                             logger=logger,
                             epoch_logging=cfg.training.log_epochs,
                             step_log_interval=cfg.training.log_interval)

    # save final state
    ckpt = {"train_state": final_state, "metrics_history": trace}
    ckpt_mgr.save(cfg.training.nepochs + 1, ckpt, force=True)

    # plot metrics traces
    # fig, _ = plot_metrics_history(trace["train"], figsize=(10, 5))
    # fig.set(title = "Training Curves")
    # fig.savefig("metrics-train.pdf")
    # fig, _ = plot_metrics_history(trace["test"], figsize=(10, 5))
    # fig.set(title = "Test Curves")
    # fig.savefig("metrics-test.pdf")

if __name__ == "__main__":
    # prevent TF from using the GPU
    tf.config.experimental.set_visible_devices([], "GPU")
    # use GPU 1 of lambda machine
    jax.config.update("jax_default_device", jax.devices()[1])
    # set seaborn config
    sns.set_theme(context="notebook",
                  font_scale=1.125,
                  style="ticks",
                  palette="Set2",
                  rc={"figure.dpi": 600})
    # run training
    main()
