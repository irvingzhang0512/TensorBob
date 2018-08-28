import tensorflow as tf
from tensorbob.dataset.dataset_utils import get_dataset_by_config

__all__ = ['BaseDataset', 'MergedDataset']


class BaseDataset:
    """
    单个 tf.data.Dataset 实例的封装
    通过输入的 dataset_configs 等条件获取 tf.data.Dataset 实例
    包装 tf.data.Dataset 实例，保存相关的 iterator, next_batch 等信息
    """

    def __init__(self, dataset_configs,
                 batch_size=32,
                 shuffle=False,
                 shuffle_buffer_size=None,
                 repeat=1,
                 prefetch_buffer_size=None
                 ):
        """
        根据 dataset_configs 获取基本的 tf.data.Dataset实例
        再通过几个参数进一步设置 dataset。

        :param dataset_configs:         list类型，元素为map，即基本 dataset 的配置文件，其他请参考 dataset_utils。
        :param batch_size:              Batch Size
        :param shuffle:                 是否需要打乱顺序
        :param shuffle_buffer_size:     默认使用 dataset.size 作为参数
        :param repeat:                  重复次数
        :param prefetch_buffer_size:    prefetch参数，若为None则不进行 prefetch 操作
        """
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._shuffle_buffer_size = shuffle_buffer_size
        self._repeat = repeat
        self._size = None

        # 根据输入的 dict 配置文件，获取一组 tf.data.Dataset 实例
        datasets = []
        for dataset_config in dataset_configs:
            if not isinstance(dataset_config, dict):
                raise ValueError('dataset_config must be dict instead of {}'.format(type(dataset_config)))
            cur_dataset, cur_dataset_size = get_dataset_by_config(dataset_config)
            datasets.append(cur_dataset)
            if self._size:
                assert self._size == cur_dataset_size
            else:
                self._size = cur_dataset_size

        # dataset 操作
        dataset = tf.data.Dataset.zip(tuple(datasets))
        dataset = dataset.repeat(self._repeat)
        if shuffle:
            if shuffle_buffer_size is None:
                shuffle_buffer_size = self.size
            dataset = dataset.shuffle(shuffle_buffer_size)
        if prefetch_buffer_size:
            dataset = dataset.prefetch(prefetch_buffer_size)
        self._tf_dataset = dataset.batch(batch_size)

        # dataset iterator 相关参数
        self._iterator = self._tf_dataset.make_initializable_iterator()
        self._next_batch = self._iterator.get_next()

    @property
    def tf_dataset(self):
        return self._tf_dataset

    @property
    def size(self):
        return self._size

    @property
    def iterator(self):
        return self._iterator

    @property
    def next_batch(self):
        return self._next_batch

    @property
    def batch_size(self):
        return self._batch_size

    def reset(self, sess, feed_dict=None):
        sess.run(self.iterator.initializer, feed_dict=feed_dict)


class MergedDataset:
    """
    封装两个 tf.data.Dataset 实例
    使用 feedable iterator 来处理这两个实例，主要用于 训练集/验证集 的使用
    """
    def __init__(self, base_dataset_1, base_dataset_2):
        """
        输入两个数据集
        要求：在使用前必须先调用
        :param base_dataset_1:
        :param base_dataset_2:
        """
        if isinstance(base_dataset_1, tf.data.Dataset):
            self._tf_dataset_1 = base_dataset_1
            self._tf_dataset_1_iterator = self._tf_dataset_1.make_initializable_iterator()
        elif isinstance(base_dataset_1, BaseDataset):
            self._tf_dataset_1 = base_dataset_1.tf_dataset
            self._tf_dataset_1_iterator = self._tf_dataset_1.iterator
        else:
            raise TypeError
        if isinstance(base_dataset_2, tf.data.Dataset):
            self._tf_dataset_2 = base_dataset_2
            self._tf_dataset_2_iterator = self._tf_dataset_2.make_initializable_iterator()
        elif isinstance(base_dataset_2, BaseDataset):
            self._tf_dataset_2 = base_dataset_2.tf_dataset
            self._tf_dataset_2_iterator = base_dataset_2.iterator
        else:
            raise TypeError

        if self._tf_dataset_1.output_types != self._tf_dataset_2.output_types \
                or self._tf_dataset_2.output_shapes != self._tf_dataset_2.output_shapes:
            raise ValueError('Two datasets must have same output types and shapes.')

        self._ph_handle = tf.placeholder(tf.string, shape=[])
        self._iterator = tf.data.Iterator.from_string_handle(
            self._ph_handle, self._tf_dataset_1.output_types, self._tf_dataset_1.output_shapes)

        self._next_batch = self._iterator.get_next()
        self._handle_strings = None

    @property
    def tf_dataset_1(self):
        return self._tf_dataset_1

    @property
    def tf_dataset_1_iterator(self):
        return self._tf_dataset_1_iterator

    @property
    def tf_dataset_2(self):
        return self._tf_dataset_2

    @property
    def tf_dataset_2_iterator(self):
        return self._tf_dataset_2_iterator

    @property
    def ph_handle(self):
        return self._ph_handle

    @property
    def next_batch(self):
        return self._next_batch

    @property
    def iterator(self):
        return self._iterator

    @property
    def handle_strings(self):
        return self._handle_strings

    def init(self, sess):
        """
        获取数据前，必须初始化
        内容：初始化handle string，以及tf_dataset_1的
        :param sess:
        """
        self._handle_strings = sess.run([self._tf_dataset_1_iterator.string_handle(),
                                         self._tf_dataset_2_iterator.string_handle()])
        sess.run(self._tf_dataset_1_iterator.initializer)

    def get_next_batch(self, sess, handle_index=None, handle_string=None):
        if self._handle_strings is None:
            raise ValueError('must set handle strings before calling this method.')
        if handle_index is None and handle_string is None:
            raise ValueError('handle_index or handle_string cannot both be None.')
        if handle_string is not None and handle_index is not None:
            raise ValueError('handle_string and handle_index cannot both be set.')
        if handle_index is not None and handle_index not in [0, 1]:
            raise ValueError('handle_index must be 0 or 1.')
        if handle_string is not None and handle_string not in self._handle_strings:
            raise ValueError('illegal handle_string is provided.')
        cur_handle_string = handle_string if handle_string else self._handle_strings[handle_index]
        return sess.run(self._next_batch, feed_dict={self._ph_handle: cur_handle_string})

