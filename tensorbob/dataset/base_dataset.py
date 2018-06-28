import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError
from .dataset_utils import get_dataset_by_config


__all__ = ['BaseDataset']


class BaseDataset:
    """
    基本的Dataset类型
    通过输入的 dataset_configs 等条件获取 tf.data.Dataset 实例
    """

    def __init__(self, dataset_configs,
                 batch_size=32,
                 shuffle=False,
                 shuffle_buffer_size=None,
                 repeat=False,
                 prefetch_buffer_size=None
                 ):
        """
        根据 dataset_configs 获取基本的 tf.data.Dataset实例
        再通过几个参数进一步设置 dataset。

        :param dataset_configs: 获取基本 dataset 的配置文件，可以参考 dataset_utils。
        :param batch_size:
        :param shuffle: 是否需要打乱顺序
        :param shuffle_buffer_size:  默认使用 dataset.size 作为参数
        :param repeat: 是否需要进行重复
        :param prefetch_buffer_size:  是否需要prefetch数据
        """
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._shuffle_buffer_size = shuffle_buffer_size
        self._repeat = repeat
        self.size = None

        datasets = []
        for dataset_config in dataset_configs:
            if not isinstance(dataset_config, dict):
                raise ValueError('dataset_config must be dict instead of {}'.format(type(dataset_config)))
            cur_dataset, cur_dataset_size = get_dataset_by_config(dataset_config)
            datasets.append(cur_dataset)
            if self.size:
                assert self.size == cur_dataset_size
            else:
                self.size = cur_dataset_size

        # dataset 操作
        dataset = tf.data.Dataset.zip(tuple(datasets))
        if shuffle:
            if shuffle_buffer_size is None:
                shuffle_buffer_size = self.size
            dataset = dataset.shuffle(shuffle_buffer_size)
        if prefetch_buffer_size:
            dataset = dataset.prefetch(prefetch_buffer_size)
        self.dataset = dataset.batch(batch_size)

        # 获取 dataset iterator 相关内容
        self._iterator_init_flag = False
        self.iterator = self.dataset.make_initializable_iterator()
        self.next_batch = self.iterator.get_next()

    def get_next_batch(self, sess, feed_dict=None):
        """
        获取数据，通过
        :param sess:
        :param feed_dict:
        :return:
        """
        if not self._iterator_init_flag:
            self.reset(sess, feed_dict)
        try:
            return sess.run(self.next_batch)
        except OutOfRangeError:
            if self._repeat:
                sess.run(self.iterator.initializer, feed_dict=feed_dict)
                return sess.run(self.next_batch)
            raise

    def reset(self, sess, feed_dict=None):
        sess.run(self.iterator.initializer, feed_dict=feed_dict)
        self._iterator_init_flag = True
