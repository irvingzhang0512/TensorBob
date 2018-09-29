import tensorflow.contrib.slim as slim
import tensorflow as tf
from tensorbob.models.layer_utils import conv2d_transpose

__all__ = ['fc_densenet']


def _bn_relu_conv_dropout(x, out_channels, keep_drop=1.0, kernel_size=[3, 3], scope='bn_relu_conv_dropout'):
    with tf.variable_scope(scope):
        x = tf.nn.relu(slim.batch_norm(x, fused=True))
        x = slim.conv2d(x, out_channels, kernel_size, activation_fn=None, normalizer_fn=None)
        if keep_drop is not None and keep_drop != 1.0:
            x = slim.dropout(x, keep_prob=keep_drop)
        return x


def _dense_block(x, num_layers, growth_rate, keep_drop, scope=None):
    with tf.variable_scope(scope):
        stack_layers = []
        for i in range(num_layers):
            cur_x = _bn_relu_conv_dropout(x, growth_rate,
                                          keep_drop=keep_drop,
                                          scope=('bn_relu_conv_dropout_%d' % (i + 1)))
            stack_layers.append(cur_x)
            x = tf.concat([x, cur_x], -1)
        stack_res = tf.concat(stack_layers, -1)
        return x, stack_res


def _transition_down(x, out_channels, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope):
        x = _bn_relu_conv_dropout(x, out_channels, keep_prob)
        x = slim.max_pool2d(x, [2, 2])
        return x


def _transition_up(x, skip_connection, out_channels, weight_decay, scope=None):
    with tf.variable_scope(scope):
        # x = slim.conv2d_transpose(x, out_channels, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
        x = conv2d_transpose(x,
                             filter_size=[3, 3, out_channels, x.get_shape()[3]],
                             output_shape=[tf.shape(x)[0],
                                           tf.shape(skip_connection)[1],
                                           tf.shape(skip_connection)[2],
                                           out_channels],
                             strides=2,
                             weight_decay=weight_decay)
        x = tf.concat([x, skip_connection], axis=-1)
        return x


def fc_densenet(x,
                num_classes,
                is_training=False,
                first_conv_out_channels=48,
                keep_prob=0.8,
                weight_decay=0.0005,
                mode="56",
                scope="fc_densenet"
                ):
    if mode not in ["56", "67", "103"]:
        raise ValueError('unknown mode {}'.format(mode))
    if mode == '56':
        num_pools = 5
        growth_rate = 12
        n_layers_per_block = 4
    elif mode == "67":
        num_pools = 5
        growth_rate = 16
        n_layers_per_block = 5
    else:
        num_pools = 5
        growth_rate = 16
        n_layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]

    if isinstance(n_layers_per_block, int):
        n_layers_per_block = [n_layers_per_block] * (2 * num_pools + 1)
    elif isinstance(n_layers_per_block, list):
        assert len(n_layers_per_block) == (2 * num_pools + 1)
    else:
        raise ValueError('n_layers_per_block must be int or list, but get {}'.format(type(n_layers_per_block)))

    with tf.variable_scope(scope + mode):
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_initializer=tf.zeros_initializer(),
                            weights_initializer=tf.keras.initializers.he_normal(),
                            ):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                                    is_training=is_training):
                    end_points = {}
                    skip_connections = []
                    n_filters = first_conv_out_channels

                    # encoder
                    x = slim.conv2d(x, first_conv_out_channels, [3, 3], activation_fn=None, scope='first_conv')
                    for i in range(num_pools):
                        x, _ = _dense_block(x,
                                            n_layers_per_block[i],
                                            growth_rate,
                                            keep_prob,
                                            scope=('denseblock_%d' % (i + 1)))
                        end_points['denseblock_%d' % (i + 1)] = x
                        n_filters += growth_rate * n_layers_per_block[i]
                        skip_connections.append(x)
                        x = _transition_down(x,
                                             n_filters,
                                             keep_prob=keep_prob,
                                             scope=('transition_down_%d' % (i + 1)))
                        end_points['transition_down_%d' % (i + 1)] = x

                    skip_connections = skip_connections[::-1]
                    _, x = _dense_block(x,
                                        n_layers_per_block[num_pools],
                                        growth_rate,
                                        keep_prob,
                                        scope=('denseblock_%d' % (num_pools + 1)))
                    end_points['denseblock_%d' % (num_pools + 1)] = x

                    # decoder
                    for i in range(num_pools):
                        n_filters_keep = growth_rate * n_layers_per_block[num_pools + i]
                        x = _transition_up(x,
                                           skip_connections[i],
                                           n_filters_keep,
                                           weight_decay,
                                           scope=('trainsition_up_%d' % (i + 1)))
                        end_points['trainsition_up_%d' % (i + 1)] = x

                        xx, x = _dense_block(x,
                                             n_layers_per_block[num_pools + i + 1],
                                             growth_rate,
                                             keep_prob,
                                             scope=('denseblock_%d' % (num_pools + 2 + i)))
                        end_points['denseblock_%d' % (num_pools + 1 + i)] = x

                    # softmax
                    x = slim.conv2d(xx,
                                    num_classes,
                                    [1, 1],
                                    activation_fn=None,
                                    scope='logits')
                    end_points['logits'] = x
                    return x, end_points
