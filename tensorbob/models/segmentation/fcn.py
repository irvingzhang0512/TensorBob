import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.vgg import vgg_16, vgg_arg_scope
from nets.resnet_v2 import resnet_arg_scope, resnet_v2_50
from tensorbob.models.layer_utils import conv2d_transpose

__all__ = ['vgg16_fcn_8s', 'resnet50_fcn_8s']


def resnet50_fcn_8s(x,
                    num_classes=1000,
                    is_training=False,
                    weight_decay=0.0005,):
    with tf.variable_scope('resnet50_fcn_8s'):
        with slim.arg_scope(resnet_arg_scope(weight_decay=weight_decay)):
            _, end_points = resnet_v2_50(x,
                                         num_classes=1001,
                                         is_training=is_training,
                                         global_pool=False,
                                         spatial_squeeze=False,
                                         )
        with tf.variable_scope('conv_transpose_1'):
            net = conv2d_transpose(end_points['resnet50_fcn_8s/resnet_v2_50/logits'],
                                   filter_size=(4, 4,
                                                end_points['resnet50_fcn_8s/resnet_v2_50/block2'].get_shape()[3],
                                                1001),
                                   output_shape=tf.shape(end_points['resnet50_fcn_8s/resnet_v2_50/block2']),
                                   strides=2,
                                   weight_decay=weight_decay)
            net = tf.add(net, end_points['resnet50_fcn_8s/resnet_v2_50/block2'])
            end_points['resnet50_fcn_8s/conv_transpose_1'] = net
        with tf.variable_scope('conv_transpose_2'):
            net = conv2d_transpose(net,
                                   filter_size=(4, 4,
                                                end_points['resnet50_fcn_8s/resnet_v2_50/block1'].get_shape()[3],
                                                net.get_shape()[3]),
                                   output_shape=tf.shape(end_points['resnet50_fcn_8s/resnet_v2_50/block1']),
                                   strides=2,
                                   weight_decay=weight_decay)
            net = tf.add(net, end_points['resnet50_fcn_8s/resnet_v2_50/block1'])
            end_points['resnet50_fcn_8s/conv_transpose_2'] = net
        with tf.variable_scope('conv_transpose_3'):
            net = conv2d_transpose(net,
                                   filter_size=(16, 16, num_classes, net.get_shape()[3]),
                                   output_shape=(tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], num_classes),
                                   strides=8,
                                   weight_decay=weight_decay)
            end_points['resnet50_fcn_8s/conv_transpose_3'] = net
        return net, end_points


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
