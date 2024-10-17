import os

# this stops jax from stealing all the memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.tree_util as jtu
import tensorflow as tf
import tensorflow_datasets as tfds
import optax
import seaborn as sns
import hydra
from omegaconf import DictConfig
from orbax.checkpoint import (CheckpointManager,
                              CheckpointManagerOptions,
                              PyTreeCheckpointer)

from projectlib.utils import setup_rngs, instantiate_optimizer
from projectlib.data import build_dataloader, load_dataset, default_data_transforms
from projectlib.hsic import hsic_bottleneck
from projectlib.models.lif import LIFMLPCell
from projectlib.training import Metrics, TrainState, create_lif_biohsic_step, fit

@hydra.main(config_path="./configs", config_name="train-lif-biohsic", version_base=None)
def main(cfg: DictConfig):
    # use specific of machine
    jax.config.update("jax_default_device", jax.devices()[cfg.gpu])

    # setup rngs
    if cfg.nmodels > 1:
        seeds = jrng.split(jrng.PRNGKey(cfg.seed), cfg.nmodels)
        rngs = jax.vmap(setup_rngs)(seeds)
    else:
        rngs = setup_rngs(cfg.seed)
    # initialize randomness
    tf.random.set_seed(cfg.seed) # deterministic data iteration

    # setup dataloaders
    data = load_dataset(cfg.data.dataset)
    train_preprocess_fn = default_data_transforms(cfg.data.dataset, "train")
    train_loader = build_dataloader(data["train"],
                                    batch_transform=train_preprocess_fn,
                                    batch_size=cfg.data.batchsize,
                                    window_shift=1)
    test_preprocess_fn = default_data_transforms(cfg.data.dataset, "test")
    test_loader = build_dataloader(data["test"],
                                   batch_transform=test_preprocess_fn,
                                   batch_size=cfg.data.test_batchsize)

    # setup model
    model = hydra.utils.instantiate(cfg.model)
    if cfg.nmodels > 1:
        splits = jax.vmap(jrng.split)(rngs["model"])
        param_key, state_key = splits[:, 0, :], splits[:, 1, :]
    else:
        param_key, state_key = jrng.split(rngs["model"])
    init_keys = {"params": param_key}

    # create optimizer
    # opt = optax.chain(instantiate_optimizer(cfg.optimizer, len(train_loader)),
    #                   optax.ema(0.999))
    opt = instantiate_optimizer(cfg.optimizer, len(train_loader))
    # create training state (initialize parameters)
    def init_state(state_keys, model_keys):
        state_init = LIFMLPCell.initialize_carry(state_keys,
                                                 batch_dims=(cfg.data.batchsize,),
                                                 sizes=(*cfg.model.features,
                                                        cfg.model.nclasses))
        dummy_input = ([jnp.ones(u.shape) for u in state_init],
                        jnp.ones((cfg.data.batchsize, *cfg.data.shape)))
        state = TrainState.from_model(model, dummy_input, opt, model_keys,
                                      model_state=state_init,
                                      apply_fn=model.lapply)
        CustomMetrics = Metrics.with_hsic(len(state.params["params"]))
        state = state.replace(metrics=CustomMetrics.empty())

        return state
    if cfg.nmodels > 1:
        train_state = jax.vmap(init_state)(state_key, init_keys)
    else:
        train_state = init_state(state_key, init_keys)
    # create training step
    loss_fn = optax.softmax_cross_entropy
    train_step = create_lif_biohsic_step(loss_fn,
                                         cfg.hsic.gamma,
                                         cfg.hsic.sigmas,
                                         cfg.training.sample_timesteps)
    @jax.jit
    def metric_step(state: TrainState, batch, _ = None):
        xs, ys = batch
        def step(carry, _):
            model_state = carry
            (model_state, ypreds), acts = state.apply_fn(state.params,
                                                         model_state,
                                                         xs,
                                                         rngs=state.rngs)

            return model_state, (ypreds, acts)

        _, (ypreds, acts) = jax.lax.scan(step,
                                         state.model_state,
                                         jnp.arange(cfg.training.sample_timesteps))
        ypreds = jtu.tree_map(lambda x: jnp.mean(x[(x.shape[0] // 2):], axis=0), ypreds)
        acts = jtu.tree_map(lambda x: jnp.mean(x[(x.shape[0] // 2):], axis=0), acts)

        hsic_losses = {k: hsic_bottleneck(xs, ys, zs, cfg.hsic.gamma, *cfg.hsic.sigmas)
                       for k, zs in acts.items()}
        # pass input through model storing local grads along the way
        loss = jnp.mean(loss_fn(ypreds, ys))
        acc = jnp.mean(jnp.argmax(ypreds, axis=-1) == jnp.argmax(ys, axis=-1))
        metrics_updates = state.metrics.single_from_model_output(
            loss=loss, accuracy=acc,
            **{f"hsic{i}": hsic_loss[0] for i, hsic_loss in enumerate(hsic_losses.values())},
            **{f"hsicx{i}": hsic_loss[1] for i, hsic_loss in enumerate(hsic_losses.values())},
            **{f"hsicy{i}": hsic_loss[2] for i, hsic_loss in enumerate(hsic_losses.values())},
        )
        metrics = state.metrics.merge(metrics_updates)
        state = state.replace(metrics=metrics)

        return state
    if cfg.nmodels > 1:
        train_step = jax.vmap(train_step, in_axes=(0, None, 0))
        metric_step = jax.vmap(metric_step, in_axes=(0, None, 0))

    # create checkpointing utility
    ckpt_opts = CheckpointManagerOptions(
        create=True,
        save_interval_steps=cfg.checkpointing.rate,
        max_to_keep=1
    )
    ckpt_path = os.sep.join([os.getcwd(), cfg.checkpointing.path])
    ckpt_mgr = CheckpointManager(ckpt_path, PyTreeCheckpointer(), ckpt_opts)

    # create logger
    logger = hydra.utils.instantiate(cfg.logger)
    logger.log_config(cfg)

    # run training
    final_state, trace = fit({"train": train_loader, "test": test_loader},
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
    ckpt_mgr.save(cfg.training.nepochs, ckpt, force=True)

    # close logger
    logger.finish()

if __name__ == "__main__":
    # prevent TF from using the GPU
    tf.config.experimental.set_visible_devices([], "GPU")

    # set seaborn config
    sns.set_theme(context="notebook",
                  font_scale=1.125,
                  style="ticks",
                  palette="Set2",
                  rc={"figure.dpi": 600})
    # run training
    main()
