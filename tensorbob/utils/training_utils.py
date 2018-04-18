import tensorflow as tf
from tensorbob.utils.variables import variable, get_or_create_global_step


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
            _mean_ops.append(tf.assign(metric_mean, metric_sum/steps_per_epoch))

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
                self._writer.add_summary(summary_string, global_step_value/self._steps_per_epoch)

            # logging 操作
            metrics_values = run_context.session.run(self._metrics_mean)
            print("epoch {}: ".format(global_step_value/self._steps_per_epoch))
            for metric_mean, metric_mean_value in zip(self._metrics_mean, metrics_values):
                print("%s %.4f" % (metric_mean.name, metric_mean_value))

            # 每个epoch对sum清零
            run_context.session.run(self._reset_ops)

    def wrap_train_op(self, train_op):
        with tf.control_dependencies(self._new_update_ops):
            with tf.control_dependencies([train_op]):
                train_op = tf.no_op()
        return train_op
