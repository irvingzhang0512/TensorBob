from tensorflow.contrib.training.python.training.training import create_train_op
import tensorflow as tf

__all__ = ['create_train_op', 'train', 'create_finetune_train_op']


def create_finetune_train_op(train_op_stage_one, train_op_stage_two, stage_one_steps, global_step=None):
    if train_op_stage_one is None or train_op_stage_two is None or stage_one_steps is None:
        raise ValueError('train_op_stage_one ,train_op_stage_two and stage_one_steps cannot be None!')
    if global_step is None:
        global_step = tf.train.get_or_create_global_step()
    return tf.cond(tf.less_equal(global_step, stage_one_steps),
                   lambda: train_op_stage_one,
                   lambda: train_op_stage_two)


def train(train_op,
          log_dir,  # 要导入的ckpt文件路径
          scaffold=None,
          hooks=None):
    scaffold = scaffold or tf.train.Scaffold()

    with tf.train.SingularMonitoredSession(hooks=hooks, scaffold=scaffold, checkpoint_dir=log_dir) as sess:
        while not sess.should_stop():
            sess.run(train_op)


