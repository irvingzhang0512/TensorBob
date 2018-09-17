import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.resnet_v2 import resnet_arg_scope, resnet_v2_50
from tensorbob.models.layer_utils import conv2d_transpose

__all__ = ['fcn_8s_vgg16', 'fcn_8s_resnet_v2_50']


def fcn_8s_resnet_v2_50(x,
                        num_classes=1000,
                        is_training=False,
                        weight_decay=0.0005, ):
    with tf.variable_scope('resnet50_fcn_8s'):
        with slim.arg_scope(resnet_arg_scope(weight_decay=weight_decay)):
            _, end_points = resnet_v2_50(x,
                                         num_classes=num_classes,
                                         is_training=is_training,
                                         global_pool=False,
                                         spatial_squeeze=False,
                                         )
        with tf.variable_scope('conv_transpose_1'):
            net = conv2d_transpose(end_points['resnet50_fcn_8s/resnet_v2_50/logits'],
                                   filter_size=(4, 4,
                                                num_classes,
                                                end_points['resnet50_fcn_8s/resnet_v2_50/logits'].get_shape()[3]),
                                   output_shape=[tf.shape(end_points['resnet50_fcn_8s/resnet_v2_50/block2'])[0],
                                                 tf.shape(end_points['resnet50_fcn_8s/resnet_v2_50/block2'])[1],
                                                 tf.shape(end_points['resnet50_fcn_8s/resnet_v2_50/block2'])[2],
                                                 num_classes],
                                   strides=2,
                                   weight_decay=weight_decay)
            pool4_conv = slim.conv2d(end_points['resnet50_fcn_8s/resnet_v2_50/block2'],
                                     num_classes,
                                     [3, 3])
            net = tf.add(net, pool4_conv)
            end_points['resnet50_fcn_8s/conv_transpose_1'] = net
        with tf.variable_scope('conv_transpose_2'):
            net = conv2d_transpose(net,
                                   filter_size=(4, 4, num_classes, net.get_shape()[3]),
                                   output_shape=[tf.shape(end_points['resnet50_fcn_8s/resnet_v2_50/block1'])[0],
                                                 tf.shape(end_points['resnet50_fcn_8s/resnet_v2_50/block1'])[1],
                                                 tf.shape(end_points['resnet50_fcn_8s/resnet_v2_50/block1'])[2],
                                                 num_classes],
                                   strides=2,
                                   weight_decay=weight_decay)
            pool3_conv = slim.conv2d(end_points['resnet50_fcn_8s/resnet_v2_50/block1'],
                                     num_classes,
                                     [3, 3])
            net = tf.add(net, pool3_conv)
            end_points['resnet50_fcn_8s/conv_transpose_2'] = net
        with tf.variable_scope('conv_transpose_3'):
            net = conv2d_transpose(net,
                                   filter_size=(16, 16, num_classes, net.get_shape()[3]),
                                   output_shape=(tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], num_classes),
                                   strides=8,
                                   weight_decay=weight_decay)
            end_points['resnet50_fcn_8s/conv_transpose_3'] = net
        return net, end_points


def fcn_8s_vgg16(x,
                 num_classes=1000,
                 is_training=False,
                 keep_prob=0.8,
                 weight_decay=0.0005, ):
    with tf.variable_scope('vgg16_fcn_8s'):
        # encoder
        # vgg16, copy from slim, with a little changes: set max_pool2d attribute padding to 'SAME'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
                with tf.variable_scope('vgg_16') as sc:
                    end_points_collection = sc.original_name_scope + '_end_points'
                    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                        outputs_collections=end_points_collection):
                        net = slim.repeat(x, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                        net = slim.max_pool2d(net, [2, 2], scope='pool1')
                        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                        net = slim.max_pool2d(net, [2, 2], scope='pool2')
                        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                        net = slim.max_pool2d(net, [2, 2], scope='pool3')
                        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                        net = slim.max_pool2d(net, [2, 2], scope='pool4')
                        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                        net = slim.max_pool2d(net, [2, 2], scope='pool5')
                        net = slim.conv2d(net, 4096, [7, 7], padding='SAME', scope='fc6')
                        net = slim.dropout(net, keep_prob, is_training=is_training, scope='dropout6')
                        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                        net = slim.dropout(net, keep_prob, is_training=is_training,  scope='dropout7')
                        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
                        end_points[sc.name + '/fc8'] = net

        # decoder
        with tf.variable_scope('conv_transpose_1'):
            net = conv2d_transpose(end_points['vgg16_fcn_8s/vgg_16/fc8'],
                                   filter_size=(4, 4,
                                                num_classes,
                                                end_points['vgg16_fcn_8s/vgg_16/fc8'].get_shape()[3]),
                                   output_shape=[tf.shape(end_points['vgg16_fcn_8s/vgg_16/pool4'])[0],
                                                 tf.shape(end_points['vgg16_fcn_8s/vgg_16/pool4'])[1],
                                                 tf.shape(end_points['vgg16_fcn_8s/vgg_16/pool4'])[2],
                                                 num_classes],
                                   strides=2,
                                   weight_decay=weight_decay)
            pool4_conv = slim.conv2d(end_points['vgg16_fcn_8s/vgg_16/pool4'],
                                     num_classes,
                                     [3, 3])
            net = tf.add(net, pool4_conv)
            end_points['vgg16_fcn_8s/conv_transpose_1'] = net
        with tf.variable_scope('conv_transpose_2'):
            net = conv2d_transpose(net,
                                   filter_size=(4, 4, num_classes, net.get_shape()[3]),
                                   output_shape=[tf.shape(end_points['vgg16_fcn_8s/vgg_16/pool3'])[0],
                                                 tf.shape(end_points['vgg16_fcn_8s/vgg_16/pool3'])[1],
                                                 tf.shape(end_points['vgg16_fcn_8s/vgg_16/pool3'])[2],
                                                 num_classes],
                                   strides=2,
                                   weight_decay=weight_decay)
            pool3_conv = slim.conv2d(end_points['vgg16_fcn_8s/vgg_16/pool3'],
                                     num_classes,
                                     [3, 3])
            net = tf.add(net, pool3_conv)
            end_points['vgg16_fcn_8s/conv_transpose_2'] = net
        with tf.variable_scope('conv_transpose_3'):
            net = conv2d_transpose(net,
                                   filter_size=(16, 16, num_classes, net.get_shape()[3]),
                                   output_shape=(tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], num_classes),
                                   strides=8,
                                   weight_decay=weight_decay)
            end_points['vgg16_fcn_8s/conv_transpose_3'] = net
        return net, end_points
