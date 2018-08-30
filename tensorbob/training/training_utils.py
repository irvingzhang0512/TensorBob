import tensorflow as tf
from tensorflow.python.training.basic_session_run_hooks import *
from tensorflow.contrib.training.python.training.training import create_train_op
from tensorflow.python.ops.control_flow_ops import with_dependencies
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
           'SummarySaverHookV2',
           'InitFnHook',
           'GlobalStepWaiterHook',
           'ProfilerHook',
           'FinalOpsHook',
           'FeedFnHook',
           'create_train_op',
           'create_train_op_v2',
           'train',
           'create_finetune_train_op',
           'ValidationDatasetEvaluationHook', ]

_USE_GLOBAL_STEP = 0


def create_train_op_v2(total_loss,
                       optimizer,
                       global_step=_USE_GLOBAL_STEP,
                       update_ops_before_loss=None,
                       update_ops_after_loss=None,
                       variables_to_train=None,
                       transform_grads_fn=None,
                       summarize_gradients=False,
                       gate_gradients=tf.train.Optimizer.GATE_OP,
                       aggregation_method=None,
                       colocate_gradients_with_ops=False,
                       check_numerics=True):
    """
    supoort update_ops after train_op
    """
    train_op = create_train_op(total_loss=total_loss,
                               optimizer=optimizer,
                               global_step=global_step,
                               update_ops=update_ops_before_loss,
                               variables_to_train=variables_to_train,
                               transform_grads_fn=transform_grads_fn,
                               summarize_gradients=summarize_gradients,
                               gate_gradients=gate_gradients,
                               aggregation_method=aggregation_method,
                               colocate_gradients_with_ops=colocate_gradients_with_ops,
                               check_numerics=check_numerics)
    if update_ops_after_loss is not None:
        final_train_op = with_dependencies([train_op], update_ops_after_loss)
    else:
        final_train_op = train_op
    return final_train_op


def create_finetune_train_op(train_op_stage_one, train_op_stage_two, stage_one_steps, global_step=None):
    """
    根据 global stesp 选择对应的train_op
    global step <= stage_one_steps 时，选择 train_op_stage_one，否则选择 train_op_srage_two
    """
    if train_op_stage_one is None or train_op_stage_two is None or stage_one_steps is None:
        raise ValueError('train_op_stage_one ,train_op_stage_two and stage_one_steps cannot be None!')
    if global_step is None:
        global_step = tf.train.get_or_create_global_step()
    return tf.cond(tf.less_equal(global_step, stage_one_steps),
                   lambda: train_op_stage_one,
                   lambda: train_op_stage_two)


def train(train_op,
          logs_dir,  # pre-trained model
          scaffold=None,
          hooks=None,  # other hooks
          max_steps=None,  # StopAtStepHook
          logging_tensors=None, logging_every_n_steps=None,  # LoggingTensorHook
          feed_fn=None,  # FeedFnHook
          summary_writer=None, summary_op=None, summary_every_n_steps=None,  # SummarySaverHook
          saver=None, save_every_n_steps=None, checkpoint_basename="model.ckpt",  # CheckpointSaverHook
          ):
    """
    通过参数设置各种 hooks, 使用 tf.train.SingularMonitoredSession 训练
    """
    if max_steps is not None and max_steps < 0:
        raise ValueError('max_steps must be positive but get {}'.format(max_steps))

    scaffold = scaffold or tf.train.Scaffold()
    all_hooks = []

    # max training steps
    if max_steps is not None:
        all_hooks.append(StopAtStepHook(max_steps))

    # logging tensors in console
    if logging_tensors is not None and logging_every_n_steps is not None:
        if logging_every_n_steps < 0:
            raise ValueError('logging_every_n_steps must be positive but get {}'.format(logging_every_n_steps))
        all_hooks.append(LoggingTensorHook(logging_tensors, logging_every_n_steps))

    # feed_dict generator
    if feed_fn is not None:
        all_hooks.append(FeedFnHook(feed_fn))

    # summary
    if summary_every_n_steps is not None:
        if summary_every_n_steps < 0:
            raise ValueError('summary_every_n_steps must be positive but get {}'.format(summary_every_n_steps))
        if summary_op is None:
            summary_op = tf.summary.merge_all()
        all_hooks.append(SummarySaverHookV2(save_steps=summary_every_n_steps,
                                            output_dir=logs_dir,
                                            summary_writer=summary_writer,
                                            summary_op=summary_op))
        logging.debug('add summary hook...')

    # save
    if save_every_n_steps is not None:
        if saver is None:
            saver = tf.train.Saver(max_to_keep=5)
        if save_every_n_steps < 0:
            raise ValueError('save_every_n_steps must be positive but get {}'.format(save_every_n_steps))
        all_hooks.append(CheckpointSaverHook(logs_dir,
                                             save_steps=save_every_n_steps,
                                             checkpoint_basename=checkpoint_basename,
                                             saver=saver,
                                             ))

    if hooks:
        all_hooks += hooks
    with tf.train.SingularMonitoredSession(hooks=all_hooks, scaffold=scaffold, checkpoint_dir=logs_dir) as sess:
        while not sess.should_stop():
            sess.run(train_op)


class ValidationDatasetEvaluationHook(tf.train.SessionRunHook):
    """
    每经过若干 steps 就在验证集上计算一次性能指标
    """

    def __init__(self,
                 merged_dataset,
                 evaluate_every_n_steps,

                 # 预测模型参数相关
                 metrics_reset_ops,
                 metrics_update_ops,
                 evaluating_feed_dict=None,

                 # 训练结束后，save/log 相关
                 saver_file_prefix=None,
                 best_metric_var_name='best_val_metric',
                 summary_op=None, summary_writer=None,
                 summary_feed_dict=None,

                 # 学习率衰减
                 shrink_learning_rate=False,
                 shrink_epochs=3,
                 shrink_by_number=10.0):
        if merged_dataset is None:
            raise ValueError('merged_dataset cannot be None!')
        if evaluate_every_n_steps is None:
            raise ValueError('evaluate_every_n_steps cannot be None!')
        if metrics_update_ops is None:
            raise ValueError('metrics_update_ops cannot be None!')
        if metrics_reset_ops is None:
            raise ValueError('metrics_reset_ops cannot be None!')
        if summary_op is not None and summary_writer is None:
            raise ValueError('summary_writer cannot be None when summary_op is not None!')
        if shrink_learning_rate and shrink_epochs < 1:
            raise ValueError('shrink epochs must be positive')

        self._summary_feed_dict = summary_feed_dict

        # 在验证集上测试模型性能
        self._merged_dataset = merged_dataset

        # 评估模型性能的函数
        self._metrics_reset_ops = metrics_reset_ops
        self._metrics_update_ops = metrics_update_ops
        self._evaluating_feed_dict = evaluating_feed_dict

        # summary验证集上的metrics
        self._summary_op = summary_op
        self._summary_writer = summary_writer

        # 每多少次在验证集上评估一次模型性能
        self._evaluate_every_n_steps = evaluate_every_n_steps

        # 验证集上的最优性能指标记录
        self._best_val_metric = tf.get_variable(best_metric_var_name,
                                                shape=[],
                                                dtype=tf.float32,
                                                initializer=tf.zeros_initializer, )
        self._ph_best_val_metric = tf.placeholder(tf.float32, [])
        self._assign_best_val_metric_op = tf.assign(self._best_val_metric, self._ph_best_val_metric)

        # 保存验证集上性能最好的模型
        self._saver_file_prefix = saver_file_prefix
        self._saver = None
        if saver_file_prefix is not None:
            self._saver = tf.train.Saver(max_to_keep=2)

        # 学习率衰减
        self._shrink_learning_rate = shrink_learning_rate
        if shrink_learning_rate:
            with tf.variable_scope('learning_rate', reuse=True):
                self._shrink_lr_cnt = 0
                self._shrink_epochs = shrink_epochs
                lr_shrink = tf.get_variable('learning_rate_shrink',
                                            shape=[],
                                            dtype=tf.float32,
                                            initializer=tf.constant_initializer(1.0),
                                            trainable=False)
                self._shrink_lr_op = tf.assign(lr_shrink, tf.multiply(lr_shrink, shrink_by_number))

    def _evaluate_on_val_set(self, sess):
        """
        在验证集上评估模型
        """

        # 重置性能指标，重置验证集
        sess.run([self._metrics_reset_ops, self._merged_dataset.tf_dataset_2_iterator.initializer])

        # 评估模型参数
        if self._evaluating_feed_dict is None:
            self._evaluating_feed_dict = {}
        self._evaluating_feed_dict[self._merged_dataset.ph_handle] = self._merged_dataset.handle_strings[1]
        cur_metrics = None
        try:
            while True:
                cur_metrics = sess.run(self._metrics_update_ops, feed_dict=self._evaluating_feed_dict)
                # logging.debug(self._evaluating_feed_dict, cur_metrics[:len(self._metrics_update_ops)])
        except OutOfRangeError:
            pass
        return cur_metrics

    def after_run(self,
                  run_context,
                  run_values):
        sess = run_context.session

        # 获取当前global step以及最佳性能指标
        cur_global_step, best_val_metric = sess.run([tf.train.get_or_create_global_step(), self._best_val_metric])

        # 判断是否需要进行模型评估
        if cur_global_step != 0 and cur_global_step % self._evaluate_every_n_steps == 0:
            logging.info('start evaluating...')
            # 评估模型，并获取当前性能指标
            cur_metric = self._evaluate_on_val_set(sess)[0]

            if self._summary_op is not None and self._summary_writer is not None:
                summary_string = sess.run(self._summary_op)
                self._summary_writer.add_summary(summary_string, cur_global_step)
                logging.debug('add summary successfully...')

            if cur_metric > best_val_metric:
                # 若当前性能指标由于最佳值，需要更新最佳值，并看情况是否需要保存模型
                sess.run(self._assign_best_val_metric_op, feed_dict={self._ph_best_val_metric: cur_metric})
                if self._saver:
                    saver_path = self._saver.save(sess, self._saver_file_prefix, global_step=cur_global_step)
                    logging.debug('saving model into {}'.format(saver_path))

                # 学习率衰减
                if self._shrink_learning_rate:
                    logging.debug('shrink_lr_cnt reset to 0')
                    self._shrink_lr_cnt = 0
            else:
                # 性能指标没有达到最优，修改学习率衰减相关变量值
                if self._shrink_learning_rate:
                    self._shrink_lr_cnt = self._shrink_lr_cnt + 1
                    logging.debug('shrink_lr_cnt add one {}'.format(self._shrink_lr_cnt))
                    if self._shrink_lr_cnt >= self._shrink_epochs:
                        logging.debug('shrink learning rate')
                        sess.run(self._shrink_lr_op)
                        self._shrink_lr_cnt = 0
                        logging.debug('shrink_lr_cnt reset to 0')
            logging.debug('cur val metrics is %.4f and best val metrics is %.4f' % (cur_metric, best_val_metric))


class SummarySaverHookV2(tf.train.SessionRunHook):
    def __init__(self,
                 save_steps,
                 output_dir=None,
                 summary_writer=None,
                 summary_op=None):
        if save_steps is None or save_steps < 0:
            raise ValueError('save_steps must be positive integer.')
        if output_dir is None and summary_writer is None:
            raise ValueError('output_dir and summary_writer cannot be both None.')

        self._save_steps = save_steps
        self._summary_op = summary_op if summary_op is not None else tf.summary.merge_all()
        self._summary_writer = summary_writer if summary_writer is not None else tf.summary.FileWriter(output_dir,
                                                                                                       tf.get_default_graph())
        self._global_step_tensor = tf.train.get_or_create_global_step()

    def after_run(self, run_context, run_values):
        sess = run_context.session
        cur_step = sess.run(self._global_step_tensor)

        if cur_step != 0 and cur_step % self._save_steps == 0:
            summary_string = sess.run(self._summary_op)
            self._summary_writer.add_summary(summary_string, global_step=cur_step)

    def end(self, session=None):
        if self._summary_writer:
            self._summary_writer.flush()


class InitFnHook(tf.train.SessionRunHook):
    def __init__(self, init_fn):
        self._init_fn = init_fn

    def after_create_session(self, session, coord):
        self._init_fn(session)
