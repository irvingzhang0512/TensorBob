import tensorflow as tf
import tensorflow.contrib.slim as slim

__all__ = ['conv2d_transpose',
           'max_pool_with_argmax',
           'unpool_2d']


def conv2d_transpose(inputs,
                     filter_size,
                     output_shape,
                     strides,
                     initializer=slim.xavier_initializer(False),
                     padding='SAME',
                     weight_decay=0.0005,
                     scope=None):
    """
    transpose conv2d
    :param inputs:
    :param filter_size:
    :param output_shape:
    :param strides:
    :param initializer:
    :param padding:
    :param weight_decay:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope, 'Conv2dTranspose'):
        filters = slim.model_variable('weights',
                                      shape=filter_size,
                                      regularizer=slim.l2_regularizer(weight_decay),
                                      initializer=initializer,
                                      )
        if isinstance(strides, int):
            strides = [1, strides, strides, 1]
        return tf.nn.conv2d_transpose(inputs,
                                      filter=filters,
                                      output_shape=output_shape,
                                      strides=strides,
                                      padding=padding)


def max_pool_with_argmax(inputs,
                         ksize=None,
                         strides=None,
                         padding='VALID',
                         tarmax=tf.int64,
                         name=None):
    """
    与unpool_2d配合使用，直接调用了tf.nn.max_pool_with_argmax
    :param inputs:
    :param ksize:
    :param strides:
    :param padding:
    :param tarmax:
    :param name:
    :return:
    """
    if ksize is None:
        ksize = [1, 2, 2, 1]
    if strides is None:
        strides = [1, 2, 2, 1]
    return tf.nn.max_pool_with_argmax(input=inputs,
                                      ksize=ksize,
                                      strides=strides,
                                      padding=padding,
                                      Targmax=tarmax,
                                      name=name)


def unpool_2d(pool,
              ind,
              stride=None,
              scope='unpool_2d'):
    """
    Adds a 2D unpooling op.
    https://arxiv.org/abs/1505.04366
    Unpooling layer after max_pool_with_argmax.
    :param pool:    max pooled output tensor
    :param ind:     argmax indices
    :param stride:  stride is the same as for the pool
    :param scope:   scope name
    :return:        unpooling tensor
    """
    if stride is None:
        stride = [1, 2, 2, 1]
    with tf.variable_scope(scope):
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                 shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2],
                            set_input_shape[3]]
        ret.set_shape(set_output_shape)
        return ret

