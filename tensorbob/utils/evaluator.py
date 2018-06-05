import tensorflow as tf
import numpy as np
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework.errors_impl import OutOfRangeError


__all__ = ['Evaluator']


class Evaluator:
    def __init__(self,
                 batch_size=32,
                 multi_crop_number=None,
                 multi_scale_list=None,
                 evaluate_image_size=None,
                 pre_trained_model_path=None,
                 with_labels=True,
                 ):
        if multi_crop_number is not None and multi_scale_list is not None:
            raise ValueError('cannot use both multi-crop and multi-scale')
        if evaluate_image_size is None and multi_scale_list is None:
            raise ValueError('evaluate image size and multi-scale list cannot both be None')
        if pre_trained_model_path is None:
            raise ValueError('pre-trained model path cannot be None.')

        self._batch_size = batch_size

        self._multi_crop_number = multi_crop_number
        self._multi_scale_list = multi_scale_list
        self._evaluation_image_size = evaluate_image_size
        self._pre_trained_model_path = pre_trained_model_path
        self._init_fn = None
        self._with_labels = with_labels

    def _get_test_dataset(self):
        """
        :return: test_dataset & ph_evaluation_image_size
        """
        raise NotImplementedError

    def _get_graph_and_feed_dict(self):
        """
        :return: ph_x, predictions, feed_dict
        """
        raise NotImplementedError

    def _get_init_fn(self):
        """
        :return: load ckpt file
        """
        raise NotImplementedError

    def evaluate(self):
        test_dataset, ph_evaluation_image_size = self._get_test_dataset()
        logging.debug('get test_dataset & ph_evalutation_image_size successfully.')

        ph_x, ph_image_size, predictions, feed_dict = self._get_graph_and_feed_dict()
        logging.debug('get ph_x, ph_image_size, predictions, feed_dict successfully.')

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
            sess.run(tf.global_variables_initializer())
            init_fn(sess)

            # 最终结果初始化
            res = None
            labels = None

            # 进行评估
            for i in range(loop_number):
                logging.debug('start evaluate no.%d loop' % (i+1))

                # dataset 准备
                reset_feed_dict = None
                if ph_evaluation_image_size is not None:
                    if self._multi_scale_list is not None:
                        cur_image_size = self._multi_scale_list[i]
                    else:
                        cur_image_size = self._evaluation_image_size
                    reset_feed_dict = {ph_evaluation_image_size: cur_image_size}
                else:
                    cur_image_size = self._evaluation_image_size
                feed_dict[ph_image_size] = cur_image_size
                test_dataset.reset(sess, reset_feed_dict)
                logging.debug('cur dataset image size is %d' % cur_image_size)
                logging.debug('test dataset reset successfully.')

                # 当前评估结果
                cur_res = None
                while True:
                    # 评估test set中每个batch
                    try:
                        dataset_res = test_dataset.get_next_batch(sess)
                        if self._with_labels:
                            cur_x, cur_y = dataset_res
                            if i == 0:
                                labels = cur_y if labels is None else np.concatenate((labels, cur_y), axis=0)
                        else:
                            cur_x = dataset_res
#                         logging.debug('cur_x.shape is {}'.format(cur_x.shape))
                        feed_dict[ph_x] = cur_x
                        cur_predictions = sess.run(predictions, feed_dict=feed_dict)
#                         logging.debug(np.mean(np.equal(np.argmax(cur_predictions, axis=1), cur_y).astype(np.float32)))
                        cur_res = cur_predictions if cur_res is None else np.concatenate((cur_res, cur_predictions),
                                                                                         axis=0)
                    except OutOfRangeError:
                        break
                res = cur_res if res is None else res + cur_res
                logging.debug('no.%d evalutaion finished.' % i)
        if self._with_labels:
            return res/loop_number, labels
        else:
            return res/loop_number
