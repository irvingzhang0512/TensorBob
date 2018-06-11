import tensorflow as tf

__all__ = ['learning_rate_exponential_decay',
           'learning_rate_steps_dict',
           'learning_rate_val_evaluation']


def learning_rate_exponential_decay(learning_rate,
                                    global_step,
                                    decay_steps,
                                    decay_rate,
                                    staircase):
    """
    指数衰减获取学习率
    :param learning_rate:   初始值
    :param global_step:     global_step
    :param decay_steps:     每经过多少步进行衰减
    :param decay_rate:      每次衰减的比例
    :param staircase:       是否是阶梯形衰减
    :return:                学习率衰减结果
    """
    with tf.variable_scope('learning_rate'):
        if global_step is None:
            global_step = tf.train.get_or_create_global_step()
        return tf.train.exponential_decay(learning_rate,
                                          global_step,
                                          decay_steps,
                                          decay_rate,
                                          staircase,
                                          'learning_rate')


def learning_rate_steps_dict(steps_to_learning_rate_dict, min_learning_rate, global_step):
    """
    根据指定的字典获取学习率，如
    {
        10000: 0.01,
        500000: 0.001,
    }
    则[0, 10000]步以内学习率为0.01，(10000, 500000]步学习率为0.001，(500000, inf)学习率为 min_learning_rate
    :param steps_to_learning_rate_dict:     学习率字典
    :param min_learning_rate:               最小学习率
    :param global_step:                     global_step
    :return:                                返回学习率
    """
    with tf.variable_scope('learning_rate'):
        if not isinstance(steps_to_learning_rate_dict, dict):
            raise ValueError('steps_to_learning_rate_dict must be dict')
        if len(steps_to_learning_rate_dict) <= 1:
            raise ValueError('steps_to_learning_rate_dict must have at least two items')
        key_list = sorted(steps_to_learning_rate_dict.keys())
        if global_step is None:
            global_step = tf.train.get_or_create_global_step()
        cases = []
        for key in key_list:
            cases.append((global_step <= key, lambda: steps_to_learning_rate_dict[key]))
        return tf.case(cases, min_learning_rate, name='learning_rate')


def learning_rate_val_evaluation(learning_rate_start):
    """
    根据验证集结果进行学习率衰减
    配合 training.py 中的 ValidationDatasetEvaluationHook 一起使用
    如果连续 shrink_epochs 次评估验证集结果没有提升，则学习率除以 shrink_by_number
    注意，学习率初始值保存在 learning_rate_start，衰减值保存在 learning_rate_shrink
    :param learning_rate_start:     学习率初始值
    :return:                        学习率结果
    """
    with tf.variable_scope('learning_rate'):
        lr = tf.get_variable('learning_rate_start',
                             shape=[],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(learning_rate_start),
                             trainable=False)
        shrink = tf.get_variable('learning_rate_shrink',
                                 shape=[],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(1.0),
                                 trainable=False)
        return tf.div(lr, shrink, 'learning_rate')
