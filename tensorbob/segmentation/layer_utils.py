import tensorflow as tf
import tensorflow.contrib.slim as slim

__all__ = ['conv2d_transpose']


def conv2d_transpose(input_layer, filter_size, output_shape, strides, weight_decay=0.0005):
    filters = tf.get_variable('weights',
                              shape=filter_size,
                              regularizer=slim.l2_regularizer(weight_decay),
                              initializer=slim.xavier_initializer(False))
    return tf.nn.conv2d_transpose(input_layer, filter=filters, output_shape=output_shape,
                                  strides=[1, strides, strides, 1], padding='SAME')
