import tensorflow as tf
from tensorflow.python.training.basic_session_run_hooks import *
from tensorflow.python.framework.errors_impl import OutOfRangeError
from tensorflow.python.platform import tf_logging as logging

__all__ = ['SecondOrStepTimer',
           'LoggingTensorHook',
           'StopAtStepHook',
           'CheckpointSaverHook',
           'StepCounterHook',
           'NanLossDuringTrainingError',
           'NanTensorHook',
           'SummarySaverHook',
           'GlobalStepWaiterHook',
           'ProfilerHook',
           'FinalOpsHook',
           'FeedFnHook',
           'TrainDatasetFeedDictHook',
           'ValidationDatasetEvaluationHook',
           'evaluate_on_single_scale']


class TrainDatasetFeedDictHook(tf.train.SessionRunHook):
    """
    通过 BaseDataset 实例设置训练数据
    要求 BaseDataset 实例必须一次返回两个
    """

    def __init__(self, dataset,
                 ph_images,
                 ph_labels):
        self._dataset = dataset
        self._ph_x = ph_images
        self._ph_y = ph_labels

    def before_run(self, run_context):
        sess = run_context.session
        cur_images, cur_labels = self._dataset.get_next_batch(sess)
        return tf.train.SessionRunArgs(
            fetches=None, feed_dict={
                self._ph_x: cur_images,
                self._ph_y: cur_labels
            }
        )


class ValidationDatasetEvaluationHook(tf.train.SessionRunHook):
    def __init__(self,
                 dataset,
                 evaluate_every_n_steps,
                 saver_file_prefix=None,
                 summary_op=None, summary_writer=None,
                 evaluate_fn=None,
                 best_metric_var_name='best_val_metric',
                 summary_feed_dict=None):
        if dataset is None:
            raise ValueError('dataset cannot be None!')
        if evaluate_fn is None:
            raise ValueError('evaluate_fn cannot be None!')
        if evaluate_every_n_steps is None:
            raise ValueError('evaluate_every_n_steps cannot be None!')
        if summary_op is not None and summary_writer is None:
            raise ValueError('summary_writer cannot be None when summary_op is not None!')

        self._summary_feed_dict = summary_feed_dict

        # 在验证集上测试模型性能
        self._dataset = dataset

        # 评估模型性能的函数
        # 要求有两个输入，分别是(sess, dataset)
        self._evaluate_fn = evaluate_fn

        # summary验证集上的metrics
        self._summary_op = summary_op
        self._summary_writer = summary_writer

        # 每多少次在验证集上评估一次模型性能
        self._evaluate_every_n_steps = evaluate_every_n_steps

        # 验证集上的最优性能指标记录
        self._best_val_metric = tf.get_variable(best_metric_var_name, [], tf.float32)
        self._ph_best_val_metric = tf.placeholder(tf.float32, [])
        self._assign_best_val_metric_op = tf.assign(self._best_val_metric, self._ph_best_val_metric)

        # 保存验证集上性能最好的模型
        self._saver_file_prefix = saver_file_prefix
        if saver_file_prefix is not None:
            self._saver = tf.train.Saver(max_to_keep=2)

    def after_run(self,
                  run_context,
                  run_values):
        sess = run_context.session
        cur_global_step, best_val_metric = sess.run([tf.train.get_or_create_global_step(), self._best_val_metric])

        if cur_global_step != 0 and cur_global_step % self._evaluate_every_n_steps == 0:
            cur_metric = self._evaluate_fn(sess,
                                           self._dataset)
            if self._summary_op is not None and self._summary_writer is not None:
                summary_string = sess.run(self._summary_op, feed_dict=self._summary_feed_dict)
                self._summary_writer.add_summary(summary_string, cur_global_step)

            if cur_metric > best_val_metric:
                sess.run(self._assign_best_val_metric_op, feed_dict={self._ph_best_val_metric: cur_metric})
                if self._saver:
                    saver_path = self._saver.save(sess, self._saver_file_prefix, global_step=cur_global_step)
                    logging.debug('saving model into {}'.format(saver_path))
            logging.debug('best val metrics is %.4f' % cur_metric)


def evaluate_on_single_scale(scale,
                             ph_images,
                             ph_labels,
                             feed_dict,
                             ph_val_image_size,
                             metrics_reset_ops,
                             metrics_update_ops,
                             main_metric):
    if scale is None or scale < 1:
        raise ValueError('scale must be positive int')

    def evaluate_fn(sess, dataset):
        if feed_dict is None:
            raise ValueError('feed_dict must not be None')
        print('evaluate val set...')
        if ph_val_image_size is not None:
            reset_feed_dict = {ph_val_image_size: scale}
        else:
            reset_feed_dict = None
        dataset.reset(sess, feed_dict=reset_feed_dict)
        sess.run(metrics_reset_ops)
        while True:
            try:
                cur_images, cur_labels = dataset.get_next_batch(sess)
                feed_dict[ph_images] = cur_images
                feed_dict[ph_labels] = cur_labels
                sess.run(metrics_update_ops, feed_dict=feed_dict)
            except OutOfRangeError:
                break
        return sess.run(main_metric, feed_dict=feed_dict)

    return evaluate_fn
