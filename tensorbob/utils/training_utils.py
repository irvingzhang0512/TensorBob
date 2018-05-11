import tensorflow as tf
from tensorflow.python.training.basic_session_run_hooks import *
from tensorflow.python.framework.errors_impl import OutOfRangeError

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
                 saver=None,
                 saver_file_prefix=None,
                 summary_op=None,
                 summary_writer=None,
                 evaluate_fn=None):
        if dataset is None:
            raise ValueError('dataset cannot be None!')
        if evaluate_fn is None:
            raise ValueError('evaluate_fn cannot be None!')
        if evaluate_every_n_steps is None:
            raise ValueError('evaluate_every_n_steps cannot be None!')
        if saver is not None and saver_file_prefix is None:
            raise ValueError('saver_file_prefix cannot be None when saver is not None!')
        if summary_op is not None and summary_writer is None:
            raise ValueError('summary_writer cannot be None when summary_op is not None!')

        # 在验证集上测试模型性能
        self._dataset = dataset

        # 评估模型性能的函数
        # 要求有两个输入，分别是(sess, dataset)
        self._evaluate_fn = evaluate_fn

        # 保存验证集上性能最好的模型
        self._saver = saver
        self._saver_file_prefix = saver_file_prefix

        # summary验证集上的metrics
        self._summary_op = summary_op
        self._summary_writer = summary_writer

        # 每多少次在验证集上评估一次模型性能
        self._evaluate_every_n_steps = evaluate_every_n_steps

        self._best_metric = 0

    def after_run(self,
                  run_context,
                  run_values):
        sess = run_context.session
        cur_global_step = sess.run(tf.train.get_or_create_global_step())
        if cur_global_step != 0 and cur_global_step % self._evaluate_every_n_steps == 0:
            print('{}/{}'.format(cur_global_step, self._evaluate_every_n_steps))
            cur_metric = self._evaluate_fn(sess,
                                           self._dataset)

            if self._summary_op is not None and self._summary_writer is not None:
                summary_string = sess.run(self._summary_op)
                self._summary_writer.add_summary(summary_string, cur_global_step)

            if cur_metric > self._best_metric:
                self._best_metric = cur_global_step
                if self._saver:
                    self._saver.save(self._saver_file_prefix, global_step=cur_global_step)


def evaluate_on_single_scale(scale,
                             ph_image_size,
                             ph_images,
                             ph_labels,
                             ph_val_image_size,
                             ph_is_training):
    def evaluate_fn(sess, dataset):
        print('evaluate val set...')
        dataset.reset(sess, feed_dict={ph_val_image_size: scale})
        sess.run(tf.get_collection('val_metrics_reset_ops'))
        while True:
            try:
                cur_images, cur_labels = dataset.get_next_batch(sess)
                sess.run(tf.get_collection('val_metrics_update_ops'),
                         feed_dict={ph_images: cur_images,
                                    ph_labels: cur_labels,
                                    ph_image_size: scale,
                                    ph_is_training: False})
            except OutOfRangeError:
                break
        return sess.run(tf.get_collection('val_metrics')[0])

    return evaluate_fn

