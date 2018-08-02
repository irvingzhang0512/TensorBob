import tensorflow as tf
import tensorflow.contrib.slim as slim

__all__ = ['conv2d_transpose']


def conv2d_transpose(inputs,
                     filter_size,
                     output_shape,
                     strides,
                     initializer=slim.xavier_initializer(False),
                     padding='SAME',
                     weight_decay=0.0005,
                     scope=None):
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
