from tensorflow.contrib.training.python.training.training import create_train_op
import tensorflow as tf


def _get_default_hooks(log_dir, scaffold,
                       summary_per_steps, summary_per_secs, save_per_steps, save_per_secs,
                       max_steps):
    all_hooks = []
    if log_dir:
        # summary hook
        if summary_per_steps or summary_per_secs:
            all_hooks.append(tf.train.SummarySaverHook(
                scaffold=scaffold, save_steps=summary_per_steps, save_secs=summary_per_secs, output_dir=log_dir))
        # save hook
        if save_per_steps or save_per_secs:
            all_hooks.append(tf.train.CheckpointSaverHook(
                checkpoint_dir=log_dir, save_steps=save_per_steps, save_secs=save_per_steps, scaffold=scaffold))

    if max_steps and max_steps > 0:
        all_hooks.append(tf.train.StopAtStepHook(last_step=max_steps))

    return all_hooks


# 1. 通过 tf.data.Dataset 来进行训练
# 2. 计算一个Epoch内某个metrics的平均数
def train(train_op,
          log_dir,
          scaffold=None,
          hooks=None,
          summary_per_steps=None,
          summary_per_secs=None,
          save_per_steps=None,
          save_per_secs=None,
          max_steps=None):
    if log_dir is None:
        if summary_per_steps or summary_per_secs:
            raise ValueError('logdir cannot be None when summary_per_steps or summary_per_secs is not None')
        if save_per_steps or save_per_secs:
            raise ValueError('logdir cannot be None when save_per_steps or save_per_secs is not None')

    scaffold = scaffold or tf.train.Scaffold()
    all_hooks = _get_default_hooks(log_dir, scaffold,
                                   summary_per_steps, summary_per_secs, save_per_steps, save_per_secs,
                                   max_steps)
    if hooks:
        all_hooks.extend(hooks)
    with tf.train.SingularMonitoredSession(hooks=all_hooks, scaffold=scaffold, checkpoint_dir=log_dir) as sess:
        while not sess.should_stop():
            sess.run(train_op)
