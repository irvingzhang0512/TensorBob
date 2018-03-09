import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError
from .dataset_utils import *


class BaseDataset:
    def __init__(self, dataset_configs, batch_size=32, shuffle=False, shuffle_buffer_size=100, repeat=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.repeat = repeat

        datasets = []
        for dataset_config in dataset_configs:
            if not isinstance(dataset_config, dict):
                raise ValueError('dataset_config must be dict instead of {}'.format(type(dataset_config)))
            datasets.append(get_dataset_by_config(dataset_config, batch_size))
        dataset = tf.data.Dataset.zip(tuple(datasets))

        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)
        self.dataset = dataset
        self.iterator = dataset.make_initializable_iterator()
        self.next_batch = self.iterator.get_next()
        self.iterator_init_flag = False

    def get_next_batch(self, sess):
        if not self.iterator_init_flag:
            sess.run(self.iterator.initializer)
            self.iterator_init_flag = True
        try:
            return sess.run(self.next_batch)
        except OutOfRangeError:
            if not self.repeat:
                raise
            sess.run(self.iterator.initializer)
            return sess.run(self.next_batch)