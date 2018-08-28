import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.vgg import vgg_16, vgg_arg_scope
from tensorbob.models.layer_utils import conv2d_transpose

__all__ = ['vgg16_fcn_8s']


def vgg16_fcn_8s(x,
                 num_classes=1000,
                 is_training=False,
                 keep_prob=0.8,
                 weight_decay=0.0005,):
    with tf.variable_scope('vgg16_fcn_8s'):
        with slim.arg_scope(vgg_arg_scope(weight_decay=weight_decay)):
            _, end_points = vgg_16(x,
                                   num_classes=1000,
                                   is_training=is_training,
                                   dropout_keep_prob=keep_prob,
                                   spatial_squeeze=False,
                                   fc_conv_padding='SAME'
                                   )
        with tf.variable_scope('conv_transpose_1'):
            net = conv2d_transpose(end_points['vgg16_fcn_8s/vgg_16/fc8'],
                                   filter_size=(4, 4,
                                                end_points['vgg16_fcn_8s/vgg_16/pool4'].get_shape()[3],
                                                1000),
                                   output_shape=tf.shape(end_points['vgg16_fcn_8s/vgg_16/pool4']),
                                   strides=2,
                                   weight_decay=weight_decay)
            net = tf.add(net, end_points['vgg16_fcn_8s/vgg_16/pool4'])
            end_points['vgg16_fcn_8s/conv_transpose_1'] = net
        with tf.variable_scope('conv_transpose_2'):
            net = conv2d_transpose(net,
                                   filter_size=(4, 4,
                                                end_points['vgg16_fcn_8s/vgg_16/pool3'].get_shape()[3],
                                                net.get_shape()[3]),
                                   output_shape=tf.shape(end_points['vgg16_fcn_8s/vgg_16/pool3']),
                                   strides=2,
                                   weight_decay=weight_decay)
            net = tf.add(net, end_points['vgg16_fcn_8s/vgg_16/pool3'])
            end_points['vgg16_fcn_8s/conv_transpose_2'] = net
        with tf.variable_scope('conv_transpose_3'):
            net = conv2d_transpose(net,
                                   filter_size=(16, 16, num_classes, net.get_shape()[3]),
                                   output_shape=(tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], num_classes),
                                   strides=8,
                                   weight_decay=weight_decay)
            end_points['vgg16_fcn_8s/conv_transpose_3'] = net
        return net, end_points
