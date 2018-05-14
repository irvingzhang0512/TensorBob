from tensorflow.contrib.training.python.training.training import create_train_op
import tensorflow as tf
from tensorflow.python.ops.control_flow_ops import with_dependencies
from .training_utils import CheckpointSaverHook, SummarySaverHook, StopAtStepHook, LoggingTensorHook, FeedFnHook

__all__ = ['create_train_op', 'create_train_op_v2', 'train', 'create_finetune_train_op']

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
        all_hooks.append(LoggingTensorHook(logging_every_n_steps, logging_every_n_steps))

    # feed_dict generator
    if feed_fn is not None:
        all_hooks.append(FeedFnHook(feed_fn))

    # summary
    if summary_every_n_steps is not None:
        if summary_every_n_steps < 0:
            raise ValueError('summary_every_n_steps must be positive but get {}'.format(summary_every_n_steps))
        all_hooks.append(SummarySaverHook(save_steps=summary_every_n_steps,
                                          output_dir=log_dir,
                                          summary_writer=summary_writer,
                                          summary_op=summary_op))

    # save
    if save_every_n_steps is not None:
        if save_every_n_steps < 0:
            raise ValueError('save_every_n_steps must be positive but get {}'.format(save_every_n_steps))
        all_hooks.append(CheckpointSaverHook(log_dir,
                                             save_steps=save_every_n_steps,
                                             checkpoint_basename=checkpoint_basename,
                                             saver=saver,
                                             ))

    if hooks:
        all_hooks.append(hooks)
    with tf.train.SingularMonitoredSession(hooks=all_hooks, scaffold=scaffold, checkpoint_dir=log_dir) as sess:
        while not sess.should_stop():
            sess.run(train_op)
