import tensorflow as tf
from nets import inception_resnet_v2, resnet_v2, inception_v3
import densenet
import tensorflow.contrib.slim as slim


def get_classifier_and_reconstruct_model(net, args, is_training, end_points):
    cur_scale = 1
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
        d_net = None
        if args.with_mse:
            with tf.variable_scope('decoder_layer'):
                input_sizes = net.get_shape().as_list()[1:3]
                d_net = net
                for i in range(4):
                    cur_scale = cur_scale * 2
                    d_net = tf.image.resize_bilinear(d_net, [input_sizes[0] * cur_scale, input_sizes[1] * cur_scale], )
                    d_net = slim.conv2d(d_net, 64, [3, 3], normalizer_fn=None, activation_fn=None)
                    d_net = slim.batch_norm(d_net, activation_fn=tf.nn.relu)

                d_net = tf.image.resize_bilinear(d_net, [args.image_height, args.image_width], )
                d_net = slim.conv2d(d_net, args.model_out_channels, [3, 3], normalizer_fn=None, activation_fn=None)
                d_net = tf.nn.tanh(d_net)
                end_points['decoder_layer'] = d_net

        with tf.variable_scope('classifier_layer'):
            net = tf.reduce_mean(net, [1, 2], name='global_pool', keep_dims=True)
            end_points['global_pool'] = net
            net = slim.dropout(net, keep_prob=args.dropout_keep_prob, scope='Dropout_1b')
            net = slim.conv2d(net, args.num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
            end_points['logits'] = net
            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            end_points['spatial_squeeze'] = net

    return d_net, net, end_points


def get_resnet_v2_50_model(x, args, is_training):
    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=args.weight_decay,
                                                   batch_norm_decay=0.997,
                                                   batch_norm_epsilon=1e-5,
                                                   batch_norm_scale=True,
                                                   activation_fn=tf.nn.relu,
                                                   use_batch_norm=True)):
        _, end_points = resnet_v2.resnet_v2_50(x,
                                               num_classes=None,
                                               is_training=is_training,
                                               global_pool=False,
                                               output_stride=None,
                                               spatial_squeeze=True,
                                               reuse=None,
                                               scope='resnet_v2_50')
        net = end_points['resnet_v2_50/block4']
        return get_classifier_and_reconstruct_model(net, args, is_training, end_points)


def get_inception_v3_model(x, args, is_training):
    with slim.arg_scope(inception_v3.inception_v3_arg_scope(weight_decay=args.weight_decay,
                                                            use_batch_norm=True,
                                                            batch_norm_decay=0.9997,
                                                            batch_norm_epsilon=0.001,
                                                            activation_fn=tf.nn.relu)):
        _, end_points = inception_v3.inception_v3(x,
                                                  num_classes=None,
                                                  is_training=is_training,
                                                  dropout_keep_prob=args.dropout_keep_prob,
                                                  min_depth=16,
                                                  depth_multiplier=1.0,
                                                  prediction_fn=tf.nn.sigmoid,
                                                  spatial_squeeze=True,
                                                  reuse=None,
                                                  create_aux_logits=True,
                                                  scope='InceptionV3',
                                                  global_pool=False)
        net = end_points['Mixed_7c']
        return get_classifier_and_reconstruct_model(net, args, is_training, end_points)


def get_inception_resnet_v2_model(x, args, is_training):
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope(
            weight_decay=args.weight_decay,
            batch_norm_decay=0.9997,
            batch_norm_epsilon=0.001,
            activation_fn=tf.nn.relu)):
        _, end_points = inception_resnet_v2.inception_resnet_v2(x,
                                                                num_classes=None,
                                                                is_training=is_training,
                                                                dropout_keep_prob=args.dropout_keep_prob,
                                                                reuse=None,
                                                                scope='InceptionResnetV2',
                                                                create_aux_logits=False,
                                                                activation_fn=tf.nn.relu)
        net = end_points['PreAuxLogits']
        return get_classifier_and_reconstruct_model(net, args, is_training, end_points)


def get_densenet_model(x, args, is_training):
    with slim.arg_scope(densenet.densenet_arg_scope(weight_decay=args.weight_decay,
                                                    batch_norm_decay=0.99,
                                                    batch_norm_epsilon=1.1e-5,
                                                    data_format='NHWC')):
        net, end_points = densenet.densenet121(x,
                                               None,
                                               is_training=is_training,
                                               with_top=False)
        net = slim.batch_norm(net, is_training=is_training, activation_fn=tf.nn.relu)
        return get_classifier_and_reconstruct_model(net, args, is_training, end_points)