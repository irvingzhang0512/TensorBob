import tensorflow as tf
import numpy as np
from tensorflow.python.framework.errors_impl import OutOfRangeError


class Evaluator:
    def __init__(self,
                 batch_size=32,
                 multi_crop_number=None,
                 multi_scale_list=None,
                 evaluate_image_size=None):
        if multi_crop_number is not None and multi_scale_list is not None:
            raise ValueError('cannot use both multi-crop and multi-scale')
        if evaluate_image_size is None and multi_scale_list is None:
            raise ValueError('evaluate image size and multi-scale list cannot both be None')

        self._batch_size = batch_size

        self._multi_crop_number = multi_crop_number
        self._multi_scale_list = multi_scale_list
        self._evaluation_image_size = evaluate_image_size

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

    def _get_scaffold(self):
        """
        :return: load ckpt file
        """
        raise NotImplementedError

    def evaluate(self):
        test_dataset, ph_evaluation_image_size = self._get_test_dataset()
        ph_x, predictions, feed_dict = self._get_graph_and_feed_dict()

        loop_number = 1
        if self._multi_scale_list is not None:
            loop_number = len(self._multi_scale_list)
        elif self._multi_crop_number is not None:
            loop_number = self._multi_crop_number

        with tf.Session() as sess:
            # 最终结果初始化
            res = None

            # 进行评估
            for i in range(loop_number):
                # dataset 准备
                reset_feed_dict = None
                if ph_evaluation_image_size is not None:
                    if self._multi_scale_list is not None:
                        cur_image_size = self._multi_scale_list[i]
                    else:
                        cur_image_size = self._evaluation_image_size
                    reset_feed_dict = {ph_evaluation_image_size: cur_image_size}
                test_dataset.reset(sess, reset_feed_dict)

                # 当前评估结果
                cur_res = None

                while True:
                    # 评估test set中每个batch
                    try:
                        feed_dict[ph_x] = test_dataset.get_next_batch(sess)
                        cur_predictions = sess.run(predictions, feed_dict=feed_dict)
                        cur_res = cur_predictions if cur_res is None else np.concatenate((cur_res, cur_predictions),
                                                                                         axis=0)
                    except OutOfRangeError:
                        break
                res = cur_res if res is None else res + cur_res
        return res / loop_number

