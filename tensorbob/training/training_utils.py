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
           'ValidationDatasetEvaluationHook',

           'create_train_op',
           'create_train_op_v2',
           'create_finetune_train_op',

           'train',
           ]
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
    global step <= stage_one_steps 时，选择 train_op_stage_one，否则选择 train_op_stage_two
    """
    if train_op_stage_one is None or train_op_stage_two is None or stage_one_steps is None:
        raise ValueError('train_op_stage_one ,train_op_stage_two and stage_one_steps cannot be None!')
    if global_step is None:
        global_step = tf.train.get_or_create_global_step()
    return tf.cond(tf.less_equal(global_step, stage_one_steps),
                   lambda: train_op_stage_one,
                   lambda: train_op_stage_two)


class ValidationDatasetEvaluationHook(tf.train.SessionRunHook):
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
                 shrink_learning_rate_flag=False,
                 shrink_epochs=3,
                 shrink_by_number=10.0):
        """
        每经过若干 steps 就在验证集上计算一次性能指标

        主过程：
        1. 获取当前 global_step 以及当前最优性能（性能指标数值越大越好，初始化最优性能指标为0）.
        2. 判断是否需要运行验证集。
        3. 如果需要运行，则初始化验证集并将性能指标清零，并通过 metrics_update_ops 来遍历验证集、计算平均性能指标。
        4. 判断当前性能指标是否优于之前最佳值，优于最佳值则更新变量。

        其他功能如下：
        1. summary 操作：
            + 启用条件：summary_op 与 summary_writer 均不为None。
            + 运行时机：在计算完性能指标后，单独运行 sess.run 来获取 summary_string 并通过 summary_writer 保存到本地。
            + 其他：运行 summary_op 时，feed_dict 为None，需要注意。
        2. save 操作：
            + 启用条件：saver_file_prefix 不为 None。
            + 当前平均性能指标优于历史最佳性能指标时，通过保存当前模型。
            + 其他：saver_file_prefix 不是目录路径，而是文件路径。
        3. 学习率衰减：
            + 启用条件：shrink_learning_rate_flag 为 True，同时使用`trainer_utils.py` 中的 learning_rate_val_evaluation。
            + 通过 tf.get_variable 获取 `learning_rate/learning_rate_shrink`，来控制衰减率。
            + 运行时机：若性能指标连续 shrink_epochs 此小于历史最优性能指标，则增加 `learning_rate/learning_rate_shrink` 的值。

        :param merged_dataset:              获取数据集
        :param evaluate_every_n_steps:      每 evaluate_every_n_steps 运行一次验证程序
        :param metrics_reset_ops:           将 metrics 参数清零
        :param metrics_update_ops:          更新并获取 metrics 参数，一般由 tf.metrics 获取
        :param evaluating_feed_dict:        在计算性能时，所需要的 feed_dict
                                            一般是训练过程中各类 placeholder，如 ph_is_training
        :param saver_file_prefix:           保存性能最优的模型名称，包括模型名，如 "./logs/val/model.ckpt"
                                            如果为None，则不保存模型
        :param best_metric_var_name:        最优性能指标的名称，使用 tf.get_variable 获取
        :param summary_op:                  不会自动创建，必须与 summary_writer 配合使用
        :param summary_writer:              不会自动创建，必须与 summary_op 配合使用
        :param summary_feed_dict:           运行 summary_op 时所需的 feed_dict
        :param shrink_learning_rate_flag:   是否使用衰减学习率
        :param shrink_epochs:               如果连续经过 shrink_epochs 数量的epoch后，模型在验证集上的性能指标没有提升，则衰减学习率
        :param shrink_by_number:            如果要衰减学习率，选件为原来的 1/shrink_by_number
        """

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
        if shrink_learning_rate_flag and shrink_epochs < 1:
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
        self._shrink_learning_rate_flag = shrink_learning_rate_flag
        if shrink_learning_rate_flag:
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
                self._summary_writer.flush()
                logging.debug('add summary successfully...')

            if cur_metric > best_val_metric:
                # 若当前性能指标由于最佳值，需要更新最佳值，并看情况是否需要保存模型
                sess.run(self._assign_best_val_metric_op, feed_dict={self._ph_best_val_metric: cur_metric})
                if self._saver:
                    saver_path = self._saver.save(sess, self._saver_file_prefix, global_step=cur_global_step)
                    logging.debug('saving model into {}'.format(saver_path))

                # 学习率衰减
                if self._shrink_learning_rate_flag:
                    logging.debug('shrink_lr_cnt reset to 0')
                    self._shrink_lr_cnt = 0
            else:
                # 性能指标没有达到最优，修改学习率衰减相关变量值
                if self._shrink_learning_rate_flag:
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
        """
        Hook to tackle summary ops.
        Difference between SummarySaverHook and SummarySaverHookV2:
            SummarySaverHook runs summary_op along with train_op(which maybe cause some errors).
            SummarySaverHookV2 runs summary_op after train_op.
        :param save_steps:         runs summary_op every `save_steps` steps.
        :param output_dir:         if summary_writer is None, then create a tf.summary.FileWriter with `output_dir`.
        :param summary_writer:     use this writer to add summary.
        :param summary_op:         if None, then create by default tf.summary.merged_all()
        """
        if save_steps is None or save_steps < 0:
            raise ValueError('save_steps must be positive integer.')
        if output_dir is None and summary_writer is None:
            raise ValueError('output_dir and summary_writer cannot be both None.')

        self._save_steps = save_steps
        self._global_step_tensor = tf.train.get_or_create_global_step()

        if summary_op is not None:
            self._summary_op = summary_op
        else:
            self._summary_op = tf.summary.merge_all()

        if summary_writer is not None:
            self._summary_writer = summary_writer
        else:
            self._summary_writer = tf.summary.FileWriter(output_dir, tf.get_default_graph())

    def after_run(self, run_context, run_values):
        sess = run_context.session
        cur_step = sess.run(self._global_step_tensor)

        if cur_step != 0 and cur_step % self._save_steps == 0:
            summary_string = sess.run(self._summary_op)
            self._summary_writer.add_summary(summary_string, global_step=cur_step)
            self._summary_writer.flush()

    def end(self, session=None):
        if self._summary_writer:
            self._summary_writer.flush()


class InitFnHook(tf.train.SessionRunHook):
    def __init__(self, init_fn):
        """
        init_fn create a init function. This function has one param - session.

            def init_fn(session):
                pass

        :param init_fn:
        """
        self._init_fn = init_fn

    def after_create_session(self, session, coord):
        self._init_fn(session)


def train(train_op,
          logs_dir,  # pre-trained model
          session_config=None,  # tf.ConfigProto()
          scaffold=None,
          hooks=None,  # other hooks
          max_steps=None,  # StopAtStepHook
          logging_every_n_steps=None, logging_tensors=None,  # LoggingTensorHook
          feed_fn=None,  # FeedFnHook
          summary_every_n_steps=None, summary_writer=None, summary_op=None,  # SummarySaverHookV2
          save_every_n_steps=None, saver=None, checkpoint_basename="model.ckpt",  # CheckpointSaverHook
          ):
    """
    通过参数设置各种 hooks, 使用 tf.train.SingularMonitoredSession 训练

    根据输入参数创建的hook
    1. StopAtStepHook：
        + 通过 max_steps 创建。
        + 用于指定最大 global_step。
    2. FeedFnHook：
        + 通过 feed_fn 创建。
        + 用于创建一个用于 sess.run 中 feed_dict 的字典，一般用于 ph_is_training 等 placeholder 参数。
        + feed_fn 函数没有形参，
    3. SummarySaverHookV2
        + 通过 summary_writer, summary_op, summary_every_n_steps 创建。
        + 具体使用规则请参数该函数说明。
        + 要求 summary_every_n_steps 不为None，其他参数会自动创建。
    4. CheckpointSaverHook
        + 通过 saver, save_every_n_steps, checkpoint_basename 创建。
        + 用于指定根据多少 save_every_n_steps 保存。
        + 要求 save_every_n_steps 必须不为None，其他参数能够自动创建。
    5. LoggingTensorHook
        + 通过 logging_tensors 和 logging_every_n_steps 创建。
        + 要求两个参数同时不为None。
        + 用于每隔若干 steps 在 console 中打印指定 logging_tensors。

    其他功能：
    1. 每次训练前，若 logs_dir 中有ckpt文件，则导入最新的。
    2. 若需要导入 fine_tune模型，则需要自己创建 scaffold 对象。
    3. 自己还可以创建 hooks。

    :param train_op:                运行该 op 用于训练
    :param logs_dir:                指定日志文件的目录，运行时会先从该文件夹下进行restore，并保存训练 summary 数据
    :param session_config:          session的配置
    :param scaffold:                用于指定一些基本参数，如 init_fn，init_op等，可用于导入 fine_tune 模型参数
    :param hooks:                   可以自己创建一些 hooks
    :param max_steps:               训练 global_step 的上限
    :param logging_every_n_steps:   是否需要在console中打印数据
    :param logging_tensors:         需要打印的数据
    :param feed_fn:                 获取训练时所需的 feed_dict，无形参
    :param summary_every_n_steps:   是否需要启用 summary 功能
    :param summary_writer:          若为空会自动创建
    :param summary_op:              若为空会自动创建
    :param save_every_n_steps:      是否需要启用 saver 功能
    :param saver:                   若为空会自动创建
    :param checkpoint_basename:     用于 saver 功能保存文件的文件名
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
    with tf.train.SingularMonitoredSession(hooks=all_hooks, scaffold=scaffold,
                                           checkpoint_dir=logs_dir, config=session_config) as sess:
        while not sess.should_stop():
            sess.run(train_op)
