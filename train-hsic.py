import os

# this stops jax from stealing all the memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
import jax.numpy as jnp
import tensorflow as tf
import optax
import seaborn as sns
import hydra
from omegaconf import DictConfig
from orbax.checkpoint import (CheckpointManager,
                              CheckpointManagerOptions,
                              PyTreeCheckpointer)

from projectlib.utils import setup_rngs, instantiate_optimizer
from projectlib.data import load_dataset, build_dataloader, default_data_transforms
from projectlib.hsic import hsic_bottleneck
from projectlib.training import Metrics, TrainState, create_hsic_step, fit

@hydra.main(config_path="./configs", config_name="train-hsic", version_base=None)
def main(cfg: DictConfig):
    # use specific of machine
    jax.config.update("jax_default_device", jax.devices()[cfg.gpu])

    # setup rngs
    rngs = setup_rngs(cfg.seed)
    # initialize randomness
    tf.random.set_seed(cfg.seed) # deterministic data iteration

    # setup dataloaders
    data = load_dataset(cfg.data.dataset)
    preprocess_fn = default_data_transforms(cfg.data.dataset)
    train_loader = build_dataloader(data["train"],
                                    batch_transform=preprocess_fn,
                                    batch_size=cfg.data.batchsize)
    test_loader = build_dataloader(data["test"],
                                   batch_transform=preprocess_fn,
                                   batch_size=cfg.data.batchsize)

    # setup model
    model = hydra.utils.instantiate(cfg.model)
    init_keys = {"params": rngs["model"]}

    # create optimizer
    opt = instantiate_optimizer(cfg.optimizer, len(train_loader))
    # create training state (initialize parameters)
    dummy_input = jnp.ones((1, *cfg.data.shape))
    train_state = TrainState.from_model(model, dummy_input, opt, init_keys,
                                        aux_state=tuple(cfg.hsic.sigmas),
                                        apply_fn=model.lapply)
    CustomMetrics = Metrics.with_hsic(len(train_state.params["params"]))
    train_state = train_state.replace(metrics=CustomMetrics.empty())
    # create training step
    loss_fn = optax.softmax_cross_entropy
    train_step = create_hsic_step(loss_fn, cfg.hsic.gamma)
    @jax.jit
    def metric_step(state: TrainState, batch, _ = None):
        xs, ys = batch
        sigmas = state.aux_state
        ypreds, acts = state.apply_fn(state.params, xs, rngs=state.rngs)
        hsic_losses = {k: hsic_bottleneck(xs, ys, zs, cfg.hsic.gamma, *sigmas)
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
