# coding=utf-8
import tensorflow as tf

__all__ = ['get_multi_gpu_model',
           'get_multi_gpu_total_loss',
           'get_multi_gpu_train_op_by_grads',
           'get_default_config_proto',
           'average_gradients',
           ]


def get_multi_gpu_train_op_by_grads(optimizer, grads,
                                    global_step=None,
                                    UPDATE_OPS_BEFORE_GRADS=tf.GraphKeys.UPDATE_OPS):
    """
    获取训练所需的 train_op

    :param optimizer:                   训练使用的优化器
    :param grads:                       get_multi_gpu_model函数的输出结果，即平均梯度值
    :param global_step:                 训练过程中的global_step，如果为None则自动获取
    :param UPDATE_OPS_BEFORE_GRADS:     apply_gradients 之前需要运行ops
    :return:
    """
    if global_step is None:
        global_step = tf.train.get_or_create_global_step()
    with tf.control_dependencies(tf.get_collection(UPDATE_OPS_BEFORE_GRADS)):
        train_op = optimizer.apply_gradients(grads, global_step=global_step)
    return train_op


def get_multi_gpu_model(inputs, model_fn, model_args,
                        loss_fn=None, loss_args=None, labels=None, optimizer=None,

                        num_of_gpus=1, gpus_ids=None,
                        ):
    """
    数据并行的单机多卡模型模版

    :param inputs:          输入数据，后续通过 tf.split 划分
    :param model_fn:        获取logits，要求第一个参数为inputs
    :param model_args:      model_fn 其他参数
    :param loss_fn:         获取损失函数值，要求必须包含三个参数 scope，logits(model_fn的输出结果) 和 labels(tf.split后的结果)
    :param loss_args:       loss_fn 其他参数
    :param labels:          标签，顺序对应inputs，
    :param optimizer:       优化器
    :param num_of_gpus:     使用的gpu数量
    :param gpus_ids:        使用的gpu的编号，使用 f.device('/gpu:%d' % cur_id) 调用
    :return:                logits, grads（平均梯度，可直接用于 optimizer.apply_gradients）
    """
    # 判断输入数据合法性
    global split_labels, partial_loss
    assert num_of_gpus == len(gpus_ids)
    if loss_fn is not None and (labels is None or optimizer is None):
        raise ValueError('labels and optimizer cannot be None when get_loss_fn is not None.')
    if model_args is None:
        model_args = {}

    with tf.device('/cpu:0'):
        split_images = tf.split(inputs, num_or_size_splits=num_of_gpus, axis=0)

        if loss_fn is not None:
            split_labels = tf.split(labels, num_or_size_splits=num_of_gpus, axis=0)
            if loss_fn is None:
                loss_args = {}

        logits_list = []
        grads_list = [] if loss_fn is not None else None

        with tf.variable_scope(tf.get_variable_scope()):
            for i, cur_id in enumerate(gpus_ids):
                with tf.device('/gpu:%d' % cur_id):
                    with tf.name_scope('part_%d' % i) as sc:
                        # 获取 logits 和 loss
                        partial_logits = model_fn(split_images[i], **model_args)
                        if loss_fn is not None:
                            partial_loss = loss_fn(scope=sc, logits=partial_logits, labels=split_labels[i],
                                                   **loss_args)

                        tf.get_variable_scope().reuse_variables()
                        logits_list.append(partial_logits)

                        if loss_fn is not None:
                            grads = optimizer.compute_gradients(partial_loss)
                            grads_list.append(grads)

        # 计算平均grads，拼接logits
        if grads_list:
            total_grads = average_gradients(grads_list)
        else:
            total_grads = None
        total_logits = tf.concat(logits_list, axis=0, name='total_logits')

        return total_logits, total_grads


def average_gradients(tower_grads):
    """
    计算平均梯度值
    :param tower_grads:     列表形式，列表中每个元素都是 optimizer.compute_gradients 的结果
    :return:                平均梯度值，用于后续 optimizer.apply_gradients
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        cur_grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)

            cur_grads.append(expanded_g)

        cur_grad = tf.concat(cur_grads, 0)
        cur_grad = tf.reduce_mean(cur_grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (cur_grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def get_multi_gpu_total_loss(num_of_gpus):
    """
    获取损失函数取值
    损失函数取值包括 regularization_loss 和 non_regularization_loss
    regularization_loss 会在创建 variable 时，通过regularizer来自动添加对应数值到 REGULARIZER_LOSS 中
    non_regularization_loss 要手动创建，如通过 tf.losses 创建，会自动添加到 LOSSES 中
    PS：每个gpu都会计算一次 non_regularization_loss，所以 non_regularization_loss 需要除以num_of_gpus
    :param num_of_gpus: 使用GPU的数量
    :return:            一个名为 "total_loss" 的 Tensor
    """
    return tf.add(tf.losses.get_regularization_loss(),
                  tf.div(tf.add_n(tf.losses.get_losses()), num_of_gpus, name='non_regularization_loss'),
                  name='total_loss')


def get_default_config_proto(allow_growth=True,
                             allow_soft_placement=True,
                             log_device_placement=True,
                             ):
    """
    GPU相关的 tf.Session 参数
    :param allow_growth:            按需占用显存（默认为占用全部显存）
    :param allow_soft_placement:    是否允许使用其他设备（比如显存不够了，自动分配到内存中运行）
    :param log_device_placement:    是否在创建Session的时候打印一些信息（每个op被分配到哪个设备）
    :return:
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = allow_growth
    config.log_device_placement = log_device_placement
    config.allow_soft_placement = allow_soft_placement
    return config
