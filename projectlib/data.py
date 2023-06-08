import tensorflow as tf
import tensorflow_datasets as tfds
from clu.preprocess_spec import PreprocessFn
from dataclasses import dataclass
from math import ceil

from projectlib.utils import maybe

@dataclass
class ToFloat:
    name: str = "image"

    def __call__(self, features):
        return {
            k: tf.cast(v, tf.float32) / 255.0 if k == self.name else v
            for k, v in features.items()
        }

@dataclass
class OneHot:
    num_classes: int
    name: str = "label"

    def __call__(self, features):
        return {
            k: tf.one_hot(v, self.num_classes) if k == self.name else v
            for k, v in features.items()
        }

def default_data_transforms(dataset):
    if dataset == "mnist":
        return PreprocessFn([ToFloat(), OneHot(10)], only_jax_types=True)
    else:
        return None

def build_dataloader(data,
                     batch_size = 1,
                     shuffle = True,
                     batch_transform = None,
                     shuffle_buffer_size = None,
                     window_shift = None):
    # convert to a tf.data.Dataset
    # we know how to convert tfds DatasetBuilder
    # everything else is treated like tensor slices
    # can always pass in a tf.data.Dataset directly
    if not isinstance(data, tf.data.Dataset):
        if isinstance(data, tfds.core.DatasetBuilder):
            data = data.as_dataset()
        else:
            data = tf.data.Dataset.from_tensor_slices(data)
    # build data loader
    if shuffle:
        buffer_size = maybe(shuffle_buffer_size, len(data))
        data = data.shuffle(buffer_size,
                            reshuffle_each_iteration=False)
    if window_shift is not None:
        data = data.window(batch_size, shift=window_shift, drop_remainder=True)
        tfm = tf.data.experimental.assert_cardinality(len(data))
        data = data.flat_map(
            lambda e: tf.data.Dataset.zip(
                tf.nest.map_structure(lambda x: x.batch(batch_size), e)
            )
        )
        data = data.apply(tfm)
    elif batch_size > 1:
        data = data.batch(batch_size)

    # possible transform the data
    if batch_transform is not None:
        data = data.map(batch_transform)

    # prefetch the next batch as the current one is being used
    return data.prefetch(2)
