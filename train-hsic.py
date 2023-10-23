import os

# this stops jax from stealing all the memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import optax
import seaborn as sns
import hydra
from omegaconf import DictConfig
from clu import metrics
from orbax.checkpoint import (CheckpointManager,
                              CheckpointManagerOptions,
                              PyTreeCheckpointer)

from projectlib.utils import setup_rngs, instantiate_optimizer, flatten, maybe
from projectlib.data import build_dataloader, default_data_transforms
from projectlib.hsic import hsic_bottleneck
from projectlib.models.chain import Chain
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
    data = tfds.load(cfg.data.dataset)
    preprocess_fn = default_data_transforms(cfg.data.dataset)
    train_loader = build_dataloader(data["train"],
                                    batch_transform=preprocess_fn,
                                    batch_size=cfg.data.batchsize)
    test_loader = build_dataloader(data["test"],
                                   batch_transform=preprocess_fn,
                                   batch_size=256)

    # setup model
    model = hydra.utils.instantiate(cfg.model)
    chain = Chain.from_model(model)
    init_keys = {"params": rngs["model"]}

    # create optimizer
    opt = instantiate_optimizer(cfg.optimizer, len(train_loader))
    # create training state (initialize parameters)
    dummy_input = jnp.ones((1, *cfg.data.shape))
    _Metrics = Metrics.with_aux(
        **{maybe(layer.name, f"hsic_layer{i}"): metrics.Average.from_output(f"hsic{i}")
           for i, layer in enumerate(chain.models)},
        **{maybe(layer.name, f"hsicx_layer{i}"): metrics.Average.from_output(f"hsicx{i}")
           for i, layer in enumerate(chain.models)},
        **{maybe(layer.name, f"hsicy_layer{i}"): metrics.Average.from_output(f"hsicy{i}")
           for i, layer in enumerate(chain.models)}
    )
    train_state = TrainState.from_model(chain, dummy_input, opt, init_keys,
                                        apply_fn=chain.get_apply_fns(),
                                        metrics=_Metrics.empty())
    # create training step
    loss_fn = optax.softmax_cross_entropy
    train_step = create_hsic_step(loss_fn, cfg.hsic.gamma, cfg.hsic.sigmas,
                                  flatten_input=chain.flatten)
    @jax.jit
    def metric_step(state: TrainState, batch, _ = None):
        xs, ys = batch
        # a single layer's forward pass
        hsic_losses = []
        hsicx_losses = []
        hsicy_losses = []
        zs = flatten(xs) if chain.flatten else xs
        for apply_fn, params in zip(state.apply_fn, state.params):
            zs = apply_fn(params, zs, rngs=state.rngs)
            hsic_terms = hsic_bottleneck(xs, ys, zs, cfg.hsic.gamma, *cfg.hsic.sigmas)
            hsic_losses.append(hsic_terms[0])
            hsicx_losses.append(hsic_terms[1])
            hsicy_losses.append(hsic_terms[2])
        # pass input through model storing local grads along the way
        loss = jnp.mean(loss_fn(zs, ys))
        acc = jnp.mean(jnp.argmax(zs, axis=-1) == jnp.argmax(ys, axis=-1))
        metrics_updates = state.metrics.single_from_model_output(
            loss=loss, accuracy=acc,
            **{f"hsic{i}": hsic_loss for i, hsic_loss in enumerate(hsic_losses)},
            **{f"hsicx{i}": hsic_loss for i, hsic_loss in enumerate(hsicx_losses)},
            **{f"hsicy{i}": hsic_loss for i, hsic_loss in enumerate(hsicy_losses)}
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
