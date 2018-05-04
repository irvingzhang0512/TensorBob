import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from tensorbob.utils.variables import variable, get_or_create_global_step


class TrainDatasetFeedDictHook(tf.train.SessionRunHook):
    """
    通过 BaseDataset 实例设置训练数据
    要求 BaseDataset 实例必须一次返回两个
    """

    def __init__(self, dataset, ph_images, ph_labels):
        self._dataset = dataset
        self._ph_images = ph_images
        self._ph_labesl = ph_labels

    def before_run(self, run_context):
        sess = run_context.session
        cur_images, cur_labels = self._dataset.get_next_batch(sess)
        return tf.train.SessionRunArgs(
            fetches=None, feed_dict={self._ph_images: cur_images, self._ph_labesl: cur_labels}
        )


class ValidationDatasetEvaluationHook(tf.train.SessionRunHook):
    def __init__(self,
                 dataset,
                 saver, saver_file_prefix,
                 summary_op, summary_writer,
                 evaluate_every_n_steps,
                 evaluate_fn):
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
        # 要求有四个输入，分别是(sess, dataset)
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
        if cur_global_step != 0 and cur_global_step % self._evaluate_every_n_steps:
            cur_metric = self._evaluate_fn(sess,
                                           self._dataset,
                                           self._metrics,
                                           self._metrics_update_ops)

            if self._summary_op and self._summary_writer:
                summary_string = sess.run(self._summary_op)
                self._summary_writer.add_summary(summary_string, cur_global_step)

            if cur_metric > self._best_metric:
                self._best_metric = cur_global_step
                if self._saver:
                    self._saver.save(self._saver_file_prefix, global_step=cur_global_step)


def evaluate_on_single_scale(scale, ph_image_size, ph_images, ph_labels):
    def evaluate_fn(sess, dataset):
        dataset.reset(sess, feed_dict={ph_image_size: scale})
        sess.run(tf.get_collection('val_metrics_reset_ops'))
        while True:
            try:
                cur_images, cur_labels = dataset.get_next_batch(sess)
                sess.run(tf.get_collection('val_metrics_update_ops'),
                         feed_dict={ph_images: cur_images, ph_labels: cur_labels})
            except OutOfRangeError:
                break
        return sess.run(tf.get_collection('val_metrics')[0])
    return evaluate_fn


class FeedFnWithSessionHook(tf.train.SessionRunHook):
    """
    获取train_op的feed_dict
    输入一个函数，
    该函数的输入为session，返回为字典（返回字典用于sess.run([train_op], feed_dict=feed_dict)）
    """

    def __init__(self, feed_fn_with_session):
        self.feed_fn = feed_fn_with_session

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            fetches=None, feed_dict=self.feed_fn(run_context.session))


class MetricsEpochMeanHook(tf.train.SessionRunHook):
    """
    计算每个epoch中，所有metrics的平均数，并进行summary和logging
    NOTE: 使用本类时，需要将 train_op 通过 wrap_train_op 进行包装
    """

    def __init__(self, metrics_dict, steps_per_epoch, log_dir, file_writer=None):
        """
        按epoch计算每个metric的平均值，并进行summary
        :param metrics_dict: 字典（key为metric对象，value为对应的update_op）
        :param steps_per_epoch: 每个epoch包含多少steps
        :param log_dir: log地址
        :param file_writer: 如果为None，则自动创建file_writer
        """
        self._steps_per_epoch = steps_per_epoch
        self._log_dir = log_dir

        self._new_update_ops = []  # 每个step，更新metric_sum
        self._reset_ops = []  # 每个epoch，将metric_sum清零
        self._metrics_mean = []  # 保存所有metric_mean的列表

        summary_merged_ops = []  # 所有metric_mean的summary操作
        _mean_ops = []  # 每个epoch，对metric_sum求平均

        if file_writer is None and log_dir:
            self._writer = tf.summary.FileWriter(log_dir)
        for metric, update_op in metrics_dict.items():
            # 定义两个参数
            metric_mean = variable('metrics/mean/' + metric.name[:-2], shape=[],
                                   dtype=tf.float32,
                                   initializer=tf.zeros_initializer,
                                   trainable=False)
            metric_sum = variable('metrics/sum/' + metric.name[:-2], shape=[],
                                  dtype=tf.float32,
                                  initializer=tf.zeros_initializer,
                                  trainable=False)

            self._metrics_mean.append(metric_mean)
            self._reset_ops.append(tf.assign(metric_sum, tf.constant(0, tf.float32)))
            summary_merged_ops.append(tf.summary.scalar(metric_mean.name, metric_mean))
            _mean_ops.append(tf.assign(metric_mean, metric_sum / steps_per_epoch))

            # 每次step后，更新参数
            if update_op is not None:
                with tf.control_dependencies([update_op]):
                    update_op = tf.assign_add(metric_sum, metric)
            else:
                update_op = tf.assign_add(metric_sum, metric)
            self._new_update_ops.append(update_op)

        with tf.control_dependencies(_mean_ops):
            self._summary_op = tf.summary.merge(summary_merged_ops)

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):  # pylint: disable=unused-argument
        # 计算global_step
        global_step = get_or_create_global_step()
        global_step_value = run_context.session.run(global_step)
        if global_step_value != 0 and global_step_value % self._steps_per_epoch == 0:
            if self._log_dir:
                # summary操作
                summary_string = run_context.session.run(self._summary_op)
                self._writer.add_summary(summary_string, global_step_value / self._steps_per_epoch)

            # logging 操作
            metrics_values = run_context.session.run(self._metrics_mean)
            print("epoch {}: ".format(global_step_value / self._steps_per_epoch))
            for metric_mean, metric_mean_value in zip(self._metrics_mean, metrics_values):
                print("%s %.4f" % (metric_mean.name, metric_mean_value))

            # 每个epoch对sum清零
            run_context.session.run(self._reset_ops)

    def wrap_train_op(self, train_op):
        with tf.control_dependencies(self._new_update_ops):
            with tf.control_dependencies([train_op]):
                train_op = tf.no_op()
        return train_op
