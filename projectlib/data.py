import tensorflow as tf
import tensorflow_datasets as tfds
from clu.preprocess_spec import PreprocessFn
from dataclasses import dataclass
from typing import Tuple, Sequence

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

@dataclass
class RandomCrop:
    size: Tuple[int, int]
    pad: Tuple[int, int] = (0, 0)
    name: str = "image"

    def crop(self, img):
        s = tf.shape(img)
        bs, h, w, c = s[0], s[1], s[2], s[3]
        img = tf.image.resize_with_crop_or_pad(img, h + self.pad[0], w + self.pad[1])

        return tf.image.random_crop(img, [bs, self.size[0], self.size[1], c])

    def __call__(self, features):
        return {
            k: self.crop(v) if k == self.name else v
            for k, v in features.items()
        }

@dataclass
class RandomHFlip:
    name: str = "image"

    def flip(self, img):
        return tf.image.random_flip_left_right(img)

    def __call__(self, features):
        return {
            k: self.flip(v) if k == self.name else v
            for k, v in features.items()
        }

@dataclass
class Standardize:
    mean: Sequence[float]
    std: Sequence[float]
    name: str = "image"

    def standardize(self, img):
        m = tf.reshape(self.mean, (1, 1, 1, -1))
        s = tf.reshape(self.std, (1, 1, 1, -1))

        return (img - m) / s

    def __call__(self, features):
        return {
            k: self.standardize(v) if k == self.name else v
            for k, v, in features.items()
        }

def default_data_transforms(dataset):
    if dataset == "mnist":
        return PreprocessFn([ToFloat(), Standardize((0.5,), (0.5,)), OneHot(10)],
                            only_jax_types=True)
    elif dataset == "fashion_mnist":
        return PreprocessFn([ToFloat(), Standardize((0.5,), (0.5,)), OneHot(10)],
                            only_jax_types=True)
    elif dataset == "cifar10":
        return PreprocessFn([RandomCrop((32, 32), (2, 2)),
                             RandomHFlip(),
                             ToFloat(),
                             Standardize((0.4914, 0.4822, 0.4465),
                                         (0.247, 0.243, 0.261)),
                             OneHot(10)], only_jax_types=True)
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
                            reshuffle_each_iteration=True)
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
