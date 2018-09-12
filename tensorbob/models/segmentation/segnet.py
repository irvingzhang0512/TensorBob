import tensorflow as tf
from tensorbob.models.layer_utils import max_pool_with_argmax, unpool_2d
import tensorflow.contrib.slim as slim

__all__ = ['segnet_vgg16']


def segnet_vgg16(inputs,
                 num_classes=1000,
                 weight_decay=0.00005,
                 is_training=False,
                 scope='segment_vgg'):
    with tf.variable_scope(scope, 'segment_vgg', [inputs]) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                with slim.arg_scope([slim.conv2d],
                                    padding='SAME',
                                    normalizer_fn=slim.batch_norm):
                    end_points_collection = sc.original_name_scope + '_end_points'
                    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                        outputs_collections=end_points_collection):
                        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                        net, index1 = max_pool_with_argmax(net, name='pool1')
                        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                        net, index2 = max_pool_with_argmax(net, name='pool2')
                        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                        net, index3 = max_pool_with_argmax(net, name='pool3')
                        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                        net, index4 = max_pool_with_argmax(net, name='pool4')
                        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                        net, index5 = max_pool_with_argmax(net, name='pool5')

                        net = unpool_2d(net, ind=index5, scope='unpool1')
                        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv6')
                        net = unpool_2d(net, ind=index4, scope='unpool2')
                        net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv7')
                        net = slim.repeat(net, 1, slim.conv2d, 256, [3, 3], scope='conv7_3')
                        net = unpool_2d(net, ind=index3, scope='unpool3')
                        net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv8')
                        net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv8_3')
                        net = unpool_2d(net, ind=index2, scope='unpool4')
                        net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv9')
                        net = slim.repeat(net, 1, slim.conv2d, 64, [3, 3], scope='conv9_2')
                        net = unpool_2d(net, ind=index1, scope='unpool5')
                        net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='conv10')
                        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')

                        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                        return net, end_points
