import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework.errors_impl import OutOfRangeError


__all__ = ['Evaluator']


class Evaluator:
    def __init__(self,
                 batch_size=32,
                 multi_crop_number=None,
                 multi_scale_list=None,
                 pre_trained_model_path=None,
                 ):
        if multi_crop_number is not None and multi_scale_list is not None:
            raise ValueError('cannot use both multi-crop and multi-scale')
        if pre_trained_model_path is None:
            raise ValueError('pre-trained model path cannot be None.')

        self._batch_size = batch_size

        self._multi_crop_number = multi_crop_number
        self._multi_scale_list = multi_scale_list

        self._pre_trained_model_path = pre_trained_model_path

        self._metrics_reset_ops_collection = 'reset_ops'

        self._init_fn = None
        self._test_dataset = None
        self._next_batch = None

    def _get_test_dataset(self):
        """
        :return: BaseDataset object
        """
        raise NotImplementedError

    def _get_metrics_and_feed_dict(self):
        """
        :return: metrics_update, feed_dict
        """
        raise NotImplementedError

    def _get_init_fn(self):
        """
        :return: load ckpt file
        """
        raise NotImplementedError

    def evaluate(self):
        self._test_dataset = self._get_test_dataset()
        self._next_batch = self._test_dataset.next_batch

        metrics_update, feed_dict = self._get_metrics_and_feed_dict()

        if feed_dict is None:
            feed_dict = {}
        init_fn = self._get_init_fn()
        logging.debug('get init_fn successfully.')

        loop_number = 1
        if self._multi_scale_list is not None:
            loop_number = len(self._multi_scale_list)
        elif self._multi_crop_number is not None:
            loop_number = self._multi_crop_number

        logging.debug('evaluation will have %d loops' % loop_number)

        with tf.Session() as sess:
            # restore vars
            init_fn(sess)

            # 最终结果初始化
            metrics = None

            # 进行评估
            for i in range(loop_number):
                logging.debug('start evaluate no.%d loop' % (i + 1))

                sess.run(self._test_dataset.iterator.initializer)
                while True:
                    # 评估test set中每个batch
                    try:
                        cur_metrics = sess.run(metrics_update, feed_dict=feed_dict)
                    except OutOfRangeError:
                        break
                if metrics is None:
                    metrics = cur_metrics
                logging.debug('no.%d evaluation finished.' % (i + 1))
        for metric in metrics:
            print(metric)
        return metrics
