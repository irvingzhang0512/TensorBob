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
           'GlobalStepWaiterHook',
           'ProfilerHook',
           'FinalOpsHook',
           'FeedFnHook',
           'TrainDatasetFeedDictHook',
           'ValidationDatasetEvaluationHook',
           'evaluate_on_single_scale',
           'create_train_op',
           'create_train_op_v2',
           'train',
           'create_finetune_train_op']

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
        final_train_op = tf.no_op()
    return final_train_op


def create_finetune_train_op(train_op_stage_one, train_op_stage_two, stage_one_steps, global_step=None):
    if train_op_stage_one is None or train_op_stage_two is None or stage_one_steps is None:
        raise ValueError('train_op_stage_one ,train_op_stage_two and stage_one_steps cannot be None!')
    if global_step is None:
        global_step = tf.train.get_or_create_global_step()
    return tf.cond(tf.less_equal(global_step, stage_one_steps),
                   lambda: train_op_stage_one,
                   lambda: train_op_stage_two)


def train(train_op,
          log_dir,  # pre-trained model
          scaffold=None,
          hooks=None,  # other hooks
          max_steps=None,  # StopAtStepHook
          logging_tensors=None, logging_every_n_steps=None,  # LoggingTensorHook
          feed_fn=None,  # FeedFnHook
          summary_writer=None, summary_op=None, summary_every_n_steps=None,  # SummarySaverHook
          saver=None, save_every_n_steps=None, checkpoint_basename="model.ckpt",  # CheckpointSaverHook
          ):
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
        all_hooks.append(SummarySaverHook(save_steps=summary_every_n_steps,
                                          output_dir=log_dir,
                                          summary_writer=summary_writer,
                                          summary_op=summary_op))

    # save
    if save_every_n_steps is not None:
        if saver is None:
            saver = tf.train.Saver(max_to_keep=5)
        if save_every_n_steps < 0:
            raise ValueError('save_every_n_steps must be positive but get {}'.format(save_every_n_steps))
        all_hooks.append(CheckpointSaverHook(log_dir,
                                             save_steps=save_every_n_steps,
                                             checkpoint_basename=checkpoint_basename,
                                             saver=saver,
                                             ))

    if hooks:
        all_hooks += hooks
    with tf.train.SingularMonitoredSession(hooks=all_hooks, scaffold=scaffold, checkpoint_dir=log_dir) as sess:
        while not sess.should_stop():
            sess.run(train_op)


class TrainDatasetFeedDictHook(tf.train.SessionRunHook):
    """
    通过 BaseDataset 实例设置训练数据
    要求 BaseDataset 实例必须一次返回两个
    """

    def __init__(self, dataset,
                 ph_x, ph_y):
        self._dataset = dataset
        self._ph_x = ph_x
        self._ph_y = ph_y

    def before_run(self, run_context):
        sess = run_context.session
        images, labels = self._dataset.get_next_batch(sess)
        return tf.train.SessionRunArgs(
            fetches=None, feed_dict={self._ph_x: images, self._ph_y: labels}
        )


class ValidationDatasetEvaluationHook(tf.train.SessionRunHook):
    def __init__(self,
                 dataset,
                 evaluate_every_n_steps,
                 saver_file_prefix=None,
                 summary_op=None, summary_writer=None,
                 evaluate_fn=None,
                 best_metric_var_name='best_val_metric',
                 summary_feed_dict=None,
                 shrink_learning_rate=False,
                 shrink_epochs=3,
                 shrink_by_number=10.0):
        if dataset is None:
            raise ValueError('dataset cannot be None!')
        if evaluate_fn is None:
            raise ValueError('evaluate_fn cannot be None!')
        if evaluate_every_n_steps is None:
            raise ValueError('evaluate_every_n_steps cannot be None!')
        if summary_op is not None and summary_writer is None:
            raise ValueError('summary_writer cannot be None when summary_op is not None!')
        if shrink_learning_rate and shrink_epochs < 1:
            raise ValueError('shrink epochs must be positive')

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
                if self._shrink_learning_rate:
                    logging.debug('shrink_lr_cnt reset to 0')
                    self._shrink_lr_cnt = 0
            else:
                if self._shrink_learning_rate:
                    self._shrink_lr_cnt = self._shrink_lr_cnt + 1
                    logging.debug('shrink_lr_cnt add one {}'.format(self._shrink_lr_cnt))
                    if self._shrink_lr_cnt >= self._shrink_epochs:
                        logging.debug('shrink learning rate')
                        sess.run(self._shrink_lr_op)
                        self._shrink_lr_cnt = 0
                        logging.debug('shrink_lr_cnt reset to 0')
            logging.debug('cur val metrics is %.4f and best val metrics is %.4f' % (cur_metric, best_val_metric))


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
