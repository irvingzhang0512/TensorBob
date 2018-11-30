# coding=utf-8
import tensorflow as tf
import tensorbob as bob
from nets import inception_resnet_v2, resnet_v2
import densenet
import pandas as pd
import numpy as np
import os
import sys
import argparse
import tensorflow.contrib.slim as slim
from tensorflow.python.framework.errors_impl import OutOfRangeError
from tensorflow.python.platform import tf_logging as logging

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.set_verbosity(logging.DEBUG)

# # 0.28954
# thresholds = np.array([0.565, 0.39, 0.55, 0.345, 0.33, 0.39, 0.33, 0.45, 0.38, 0.39,
#                        0.34, 0.42, 0.31, 0.38, 0.49, 0.50, 0.38, 0.43, 0.46, 0.40,
#                        0.39, 0.505, 0.37, 0.47, 0.41, 0.545, 0.32, 0.1])

# # 0.3119
# thresholds = np.array([0.407, 0.441, 0.161, 0.145, 0.299, 0.129, 0.25, 0.414, 0.01, 0.028, 0.021, 0.125,
#                        0.113, 0.387, 0.602, 0.001, 0.137, 0.199, 0.176, 0.25, 0.095, 0.29, 0.159, 0.255,
#                        0.231, 0.363, 0.117, 0., ])


# # 0.2533
# thresholds = np.array([0.84716478, 0.3, 0.69826551, 0.3, 0.46891261, 0.71547698,
#                     0.35203469, 0.3, 0.3, 0.3, 0.3, 0.3,
#                     0.57338225, 0.3, 0.3, 0.3, 0.47331554, 0.3,
#                     0.56697799, 0.3, 0.3, 0.8903936, 0.35883923, 0.82715143,
#                     0.76230821, 0.78992662, 0.67384923, 0.3])

# # 0.3094
# thresholds = 0.2
#
# # 0.2791
# thresholds = 0.5

# 0.3100
thresholds = 0.15


def create_dataset(args):
    with tf.variable_scope('preprocessing'):
        def _parse_rgby_images(base_file_name):
            r_img = tf.image.decode_png(tf.read_file(base_file_name + '_red.png'), channels=1)
            g_img = tf.image.decode_png(tf.read_file(base_file_name + '_green.png'), channels=1)
            b_img = tf.image.decode_png(tf.read_file(base_file_name + '_blue.png'), channels=1)
            y_img = tf.image.decode_png(tf.read_file(base_file_name + '_yellow.png'), channels=1)
            # r_img += tf.cast((y_img / 2), tf.uint8)
            # g_img += tf.cast((y_img / 2), tf.uint8)
            img = tf.concat((r_img, g_img, b_img, y_img), axis=2)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize_images(img, (args.image_height, args.image_width))
            return img * 2.0 - 1.0

        def _image_augumentation(image):
            image = (image + 1.0) / 2.0
            # image = tf.image.random_brightness(image, max_delta=32. / 255.)
            # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            # image = tf.image.random_hue(image, max_delta=0.2)
            # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.rot90(image, tf.random_uniform([], maxval=4, dtype=tf.int32))
            return image * 2.0 - 1.0

        def _get_label_ndarays(label_strs):
            cur_labels = []
            for label_str in label_strs:
                res = np.zeros(args.num_classes, dtype=np.int32)
                res[[int(cur_label) for cur_label in label_str.split()]] = 1
                cur_labels.append(res)
            return np.stack(cur_labels, axis=0)

        if args.mode == 'train':
            csv_file_path = os.path.join(args.data_root_path, args.train_csv_file_name)
            df = pd.read_csv(csv_file_path)
            image_names = df['Id']
            image_labels = df['Target']

            # shuffle
            ids = np.arange(len(image_names))
            np.random.shuffle(ids)
            raw_image_names = image_names[ids]
            raw_image_labels = image_labels[ids]

            # train set
            label_dataset = tf.data.Dataset.from_tensor_slices(_get_label_ndarays(raw_image_labels[:-args.val_size]))
            image_names = [os.path.join(args.data_root_path, args.mode, image_name)
                           for image_name in raw_image_names[:-args.val_size]]
            image_dataset = tf.data.Dataset.from_tensor_slices(image_names)\
                .map(_parse_rgby_images).map(_image_augumentation)
            train_set = bob.dataset.BaseDataset(dataset=tf.data.Dataset.zip((image_dataset, label_dataset)),
                                                dataset_size=len(image_names) - args.val_size,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                shuffle_buffer_size=args.shuffle_buffer_size,
                                                repeat=args.epochs,
                                                )

            # val set
            label_dataset = tf.data.Dataset.from_tensor_slices(_get_label_ndarays(raw_image_labels[-args.val_size:]))
            image_names = [os.path.join(args.data_root_path, args.mode, image_name) for image_name in
                           raw_image_names[-args.val_size:]]
            image_dataset = tf.data.Dataset.from_tensor_slices(image_names).map(_parse_rgby_images)
            val_set = bob.dataset.BaseDataset(dataset=tf.data.Dataset.zip((image_dataset, label_dataset)),
                                              dataset_size=args.val_size,
                                              batch_size=args.batch_size, )

            return bob.dataset.MergedDataset(train_set, val_set)
        elif args.mode == 'test':
            csv_file_path = os.path.join(args.data_root_path, args.submission_csv_file_name)
            df = pd.read_csv(csv_file_path)
            image_names = df['Id']
            image_names = [os.path.join(args.data_root_path, args.mode, image_name) for image_name in image_names]
            image_dataset = tf.data.Dataset.from_tensor_slices(image_names).map(_parse_rgby_images)
            file_name_dataset = tf.data.Dataset.from_tensor_slices(df['Id'])
            return bob.dataset.BaseDataset(dataset=tf.data.Dataset.zip((image_dataset, file_name_dataset)),
                                           dataset_size=len(image_names),
                                           batch_size=args.batch_size, )
        else:
            csv_file_path = os.path.join(args.data_root_path, args.train_csv_file_name)
            df = pd.read_csv(csv_file_path)
            image_names = df['Id']
            image_labels = df['Target']

            label_dataset = tf.data.Dataset.from_tensor_slices(_get_label_ndarays(image_labels))
            image_names = [os.path.join(args.data_root_path, 'train', image_name)
                           for image_name in image_names]
            image_dataset = tf.data.Dataset.from_tensor_slices(image_names).map(_parse_rgby_images)
            return bob.dataset.BaseDataset(dataset=tf.data.Dataset.zip((image_dataset, label_dataset)),
                                           dataset_size=len(image_names) - args.val_size,
                                           batch_size=args.batch_size, )


def get_model(x, args, is_training=False, ):
    # with slim.arg_scope(densenet.densenet_arg_scope(weight_decay=args.weight_decay,
    #                                                 batch_norm_decay=0.99,
    #                                                 batch_norm_epsilon=1.1e-5,
    #                                                 data_format='NHWC')):
    #     return densenet.densenet121(x,
    #                                 num_classes=args.num_classes,
    #                                 is_training=is_training, )
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope(
            weight_decay=args.weight_decay,
            batch_norm_decay=0.9997,
            batch_norm_epsilon=0.001,
            activation_fn=tf.nn.relu)):
        return inception_resnet_v2.inception_resnet_v2(x,
                                                       num_classes=args.num_classes,
                                                       is_training=is_training,
                                                       dropout_keep_prob=args.dropout_keep_prob,
                                                       reuse=None,
                                                       scope='InceptionResnetV2',
                                                       create_aux_logits=False,
                                                       activation_fn=tf.nn.relu)


def get_encoder_decoder_model(x, args, is_training=False, in_channels=3, out_channels=1, ):
    if in_channels == 3:
        channels = tf.split(axis=3, num_or_size_splits=4, value=x)
        x = tf.concat(axis=3, values=channels[:3])
    # with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=args.weight_decay,
    #                                                batch_norm_decay=0.997,
    #                                                batch_norm_epsilon=1e-5,
    #                                                batch_norm_scale=True,
    #                                                activation_fn=tf.nn.relu,
    #                                                use_batch_norm=True)):
    #     net, end_points = resnet_v2.resnet_v2_101(x,
    #                                               num_classes=None,
    #                                               is_training=is_training,
    #                                               global_pool=False,
    #                                               output_stride=None,
    #                                               spatial_squeeze=True,
    #                                               reuse=None,
    #                                               scope='resnet_v2_101')
    # with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope(
    #         weight_decay=args.weight_decay,
    #         batch_norm_decay=0.9997,
    #         batch_norm_epsilon=0.001,
    #         activation_fn=tf.nn.relu)):
    #     _, end_points = inception_resnet_v2.inception_resnet_v2(x,
    #                                                             num_classes=None,
    #                                                             is_training=is_training,
    #                                                             dropout_keep_prob=args.dropout_keep_prob,
    #                                                             reuse=None,
    #                                                             scope='InceptionResnetV2',
    #                                                             create_aux_logits=False,
    #                                                             activation_fn=tf.nn.relu)
    #     net = end_points['PreAuxLogits']

    with slim.arg_scope(densenet.densenet_arg_scope(weight_decay=args.weight_decay,
                                                    batch_norm_decay=0.99,
                                                    batch_norm_epsilon=1.1e-5,
                                                    data_format='NHWC')):
        net, end_points = densenet.densenet121(x,
                                               None,
                                               is_training=is_training,
                                               with_top=False)
        net = slim.batch_norm(net, is_training=is_training, activation_fn=tf.nn.relu)

        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with tf.variable_scope('decoder_layer'):
                input_sizes = net.get_shape().as_list()[1:3]
                d_net = net
                for i in range(4):
                    cur_scale = (i + 1) * 2
                    d_net = tf.image.resize_bilinear(d_net, [input_sizes[0] * cur_scale, input_sizes[1] * cur_scale], )
                    d_net = slim.conv2d(d_net, 64, [3, 3], normalizer_fn=None, activation_fn=None)
                    d_net = slim.batch_norm(d_net, activation_fn=tf.nn.relu)

                d_net = tf.image.resize_bilinear(d_net, [args.image_height, args.image_width], )
                d_net = slim.conv2d(d_net, out_channels, [3, 3], normalizer_fn=None, activation_fn=None)
                d_net = tf.nn.tanh(d_net)
                end_points['decoder_layer'] = d_net

            with tf.variable_scope('classifier_layer'):
                net = tf.reduce_mean(net, [1, 2], name='global_pool', keep_dims=True)
                end_points['global_pool'] = net
                net = slim.batch_norm(net, activation_fn=tf.nn.relu)
                net = slim.conv2d(net, args.num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
                end_points['logits'] = net
                net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                end_points['spatial_squeeze'] = net

        return d_net, net, end_points


def get_classifier_and_reconstruction_loss(args,
                                           classifier_logits, classifier_labels,
                                           reconstruct_image, raw_images, out_channels=4,
                                           ):
    # classifier loss
    classifier_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=classifier_labels, logits=classifier_logits)
    # f1_loss(classifier_labels, tf.sigmoid(classifier_logits), thresholds)
    # focal_loss(classifier_logits, classifier_labels)

    # reconstruction loss
    if out_channels == 3:
        channels = tf.split(axis=3, num_or_size_splits=4, value=raw_images)
        raw_images = tf.concat(axis=3, values=channels[:3])
    elif out_channels == 1:
        channels = tf.split(axis=3, num_or_size_splits=4, value=raw_images)
        raw_images = channels[3]

    reconstruction_loss = tf.losses.mean_squared_error(predictions=reconstruct_image, labels=raw_images)

    seperate_loss_summary_op = tf.summary.merge([
        tf.summary.scalar('classifier_loss', classifier_loss),
        tf.summary.scalar('reconstruction_loss', reconstruction_loss)
    ])

    return tf.losses.get_total_loss(), seperate_loss_summary_op


def focal_loss(logits, labels, alpha=0.25, gamma=2):
    with tf.variable_scope('focal_loss'):
        sigmoid_p = tf.nn.sigmoid(logits)
        cur_labels = tf.cast(labels, tf.float32)

        pt = sigmoid_p * cur_labels + (1 - sigmoid_p) * (1 - cur_labels)
        w = alpha * cur_labels + (1 - alpha) * (1 - cur_labels)
        w = w * tf.pow((1 - pt), gamma)

        return tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits, weights=w,
                                               scope='focal_loss_partial')


def f1_loss(y_true, y_pred, threshold):
    with tf.variable_scope('f1_loss'):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.greater(tf.cast(y_pred, tf.float32), threshold), tf.float32)
        tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32), axis=0)
        tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.float32), axis=0)
        fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32), axis=0)
        fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32), axis=0)

        p = tp / (tp + fp + 1e-7)
        r = tp / (tp + fn + 1e-7)

        f1 = 2 * p * r / (p + r + 1e-7)
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        cur_loss = 1 - tf.reduce_mean(f1)
        tf.losses.add_loss(cur_loss)
        return cur_loss


def get_loss(logits, labels, args):
    tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
    f1_loss(labels, tf.sigmoid(logits), thresholds)
    #     focal_loss(logits, labels)
    return tf.losses.get_total_loss()


def fbeta_score_macro(y_true, y_pred, beta=1, threshold=thresholds):
    # https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.greater(tf.cast(y_pred, tf.float32), threshold), tf.float32)

    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)

    f1 = (1 + beta ** 2) * p * r / ((beta ** 2) * p + r + 1e-7)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return tf.reduce_mean(f1)


def get_metrics(total_loss, logits, labels, args):
    f1 = fbeta_score_macro(labels, tf.sigmoid(logits), threshold=thresholds)
    summary_loss, update_loss = tf.metrics.mean(total_loss, name='loss')
    summary_f1, update_f1 = tf.metrics.mean(f1, name='f1')

    for metric in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES):
        tf.add_to_collection('RESET_OPS',
                             tf.assign(metric, tf.zeros(metric.get_shape(), metric.dtype)))

    with tf.control_dependencies(tf.get_collection('RESET_OPS')):
        after_reset_loss = tf.identity(update_loss, 'logging_loss')
        after_reset_f1 = tf.identity(update_f1, 'logging_f1')

    summary_loss = tf.summary.scalar('loss', summary_loss)
    summary_f1 = tf.summary.scalar('f1', summary_f1)
    summary_metrics = tf.summary.merge([summary_loss, summary_f1])

    return [summary_f1, summary_loss], [update_f1, update_loss], [after_reset_f1, after_reset_loss], summary_metrics


def main(args):
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    else:
        raise ValueError('unknown mode {}'.format(args.mode))


def evaluate(args):
    # 构建数据集
    dataset = create_dataset(args)
    input_images, input_labels = dataset.next_batch

    # 构建模型
    ph_is_training = tf.placeholder(tf.bool)
    reconstruct_image, logits, end_points = get_encoder_decoder_model(input_images, args,
                                                                      is_training=ph_is_training,
                                                                      in_channels=3,
                                                                      out_channels=1, )

    n_sigmoid = tf.nn.sigmoid(logits)
    f1_score = fbeta_score_macro(input_labels, n_sigmoid, threshold=thresholds)

    with tf.Session() as sess:
        # 初始化模型
        sess.run(tf.global_variables_initializer())
        sess.run(dataset.iterator.initializer)
        saver = tf.train.Saver()
        saver.restore(sess, args.trained_model)

        i = 0
        total_f1 = .0
        try:
            while True:
                cur_f1 = sess.run(f1_score, feed_dict={ph_is_training: False})
                total_f1 = total_f1 + cur_f1
                i += 1
                if i % 100 == 0:
                    print('%d: cur f1 score: %.4f' % (i, cur_f1))
        except OutOfRangeError:
            pass
    print('final f1: %.4f' % (total_f1 / i))


def test(args):
    # 构建数据集
    dataset = create_dataset(args)
    input_images, input_file_names = dataset.next_batch

    # 构建模型
    ph_is_training = tf.placeholder(tf.bool)

    # logits, end_points = get_model(input_images, args, ph_is_training)
    reconstruct_image, logits, end_points = get_encoder_decoder_model(input_images, args,
                                                                      is_training=ph_is_training,
                                                                      in_channels=3,
                                                                      out_channels=1, )

    predictions = tf.cast(tf.sigmoid(logits) >= thresholds, tf.int32)

    df = pd.DataFrame(columns=('Id', 'Predicted'))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(dataset.iterator.initializer)
        saver = tf.train.Saver()
        saver.restore(sess, args.trained_model)

        try:
            while True:
                cur_predictions, cur_file_names = sess.run([predictions, input_file_names],
                                                           feed_dict={ph_is_training: False})
                for cur_predictions, cur_file_name in zip(cur_predictions, cur_file_names):
                    label_str = " ".join([str(cur_num) for cur_num in np.where(cur_predictions == 1)[0]])
                    if label_str == '' or label_str is None:
                        label_str = "0"
                    df = df.append({'Id': cur_file_name.decode(), 'Predicted': label_str}, True)
        except OutOfRangeError:
            pass

        from datetime import datetime
        file_name = "./res_{}.csv".format(datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M"))
        df.to_csv(file_name, index=False)


def train(args):
    # 构建数据集
    dataset = create_dataset(args)
    input_images, input_labels = dataset.next_batch

    # 构建模型
    ph_is_training = tf.placeholder(tf.bool)

    # # multi-label classification
    # logits, end_points = get_model(input_images, args, ph_is_training)
    # total_loss = get_loss(logits, input_labels, args)

    # multi-label classification & reconstruction
    reconstruct_image, logits, end_points = get_encoder_decoder_model(input_images, args,
                                                                      is_training=ph_is_training,
                                                                      in_channels=3,
                                                                      out_channels=1, )
    total_loss, seperate_loss_summary = get_classifier_and_reconstruction_loss(args,
                                                                               logits, input_labels,
                                                                               reconstruct_image, input_images,
                                                                               out_channels=1, )

    lr = bob.training.learning_rate_exponential_decay(args.learning_rate_start,
                                                      tf.train.get_or_create_global_step(),
                                                      args.learning_rate_decay_steps,
                                                      args.learning_rate_decay_rate,
                                                      args.learning_rate_staircase)
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
    train_op = bob.training.create_train_op(total_loss,
                                            optimizer,
                                            global_step=tf.train.get_or_create_global_step())

    # 构建metrics
    summary_metrics, update_metrics, after_reset_metrics, metrics_summary_op = get_metrics(total_loss, logits,
                                                                                           input_labels, args)
    training_summary_writer = tf.summary.FileWriter(args.logs_dir, tf.get_default_graph())

    # 构建hooks
    summary_hook = bob.training.SummarySaverHook(summary_op=seperate_loss_summary,
                                                 save_steps=args.summary_every_n_steps,
                                                 summary_writer=training_summary_writer,)
    val_feed_dict = {ph_is_training: False}
    validation_hook = bob.training.ValidationDatasetEvaluationHook(dataset,
                                                                   evaluate_every_n_steps=args.validation_every_n_steps,

                                                                   metrics_reset_ops=tf.get_collection('RESET_OPS'),
                                                                   metrics_update_ops=update_metrics,
                                                                   evaluating_feed_dict=val_feed_dict,

                                                                   summary_op=metrics_summary_op,
                                                                   summary_writer=tf.summary.FileWriter(
                                                                       args.val_logs_dir,
                                                                       tf.get_default_graph()),

                                                                   saver_file_prefix=os.path.join(args.val_logs_dir,
                                                                                                  'model.ckpt'),
                                                                   )
    init_fn_hook = bob.training.InitFnHook(dataset.init)
    hooks = [validation_hook, init_fn_hook, summary_hook]

    # 其他
    def feed_fn():
        return {dataset.ph_handle: dataset.handle_strings[0],
                ph_is_training: True}

    # scaffold
    if args.pretrained_model_path is None:
        scaffold = None
    else:
        var_list = bob.variables.get_variables_to_restore(include=args.pretrained_model_includes,
                                                          exclude=args.pretrained_model_excludes)
        init_fn = bob.variables.assign_from_checkpoint_fn(args.pretrained_model_path,
                                                          var_list=var_list,
                                                          ignore_missing_vars=True,
                                                          reshape_variables=False)

        def scaffold_init_fn(_, session):
            init_fn(session)

        scaffold = tf.train.Scaffold(init_fn=scaffold_init_fn)

    # 训练
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    bob.training.train(train_op, args.logs_dir,
                       session_config=config,
                       hooks=hooks,
                       scaffold=scaffold,
                       logging_tensors=after_reset_metrics,
                       logging_every_n_steps=args.logging_every_n_steps,
                       feed_fn=feed_fn,
                       summary_every_n_steps=args.summary_every_n_steps,
                       summary_op=metrics_summary_op,
                       summary_writer=training_summary_writer,
                       save_every_n_steps=args.save_every_n_steps,
                       )


def _parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # base configs
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--num_classes', type=int, default=28)
    # parser.add_argument('--data_root_path', type=str, default="/home/tensorflow05/data/kaggle/protein")
    parser.add_argument('--data_root_path', type=str, default="/ssd/zhangyiyang/protein")
    parser.add_argument('--train_csv_file_name', type=str, default="train.csv")
    parser.add_argument('--submission_csv_file_name', type=str, default="sample_submission.csv")
    parser.add_argument('--image_height', type=int, default=512)
    parser.add_argument('--image_width', type=int, default=512)

    # training base configs
    parser.add_argument('--val_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=.0001)
    parser.add_argument('--dropout_keep_prob', type=float, default=0.8)
    parser.add_argument('--shuffle_buffer_size', type=int, default=1000)
    parser.add_argument('--pretrained_model_includes', type=list, default=['InceptionResnetV2'])
    parser.add_argument('--pretrained_model_excludes', type=list, default=None)
    # parser.add_argument('--pretrained_model_path', type=str,
    #                     default="/ssd/zhangyiyang/slim/resnet_v2_101_2017_04_14/resnet_v2_101.ckpt")
    # parser.add_argument('--pretrained_model_path', type=str,
    #                     default="/home/tensorflow05/data/pre-trained/slim/inception_resnet_v2_2016_08_30.ckpt")
    # parser.add_argument('--pretrained_model_path', type=str,
    #                     default="/ssd/zhangyiyang/slim/inception_resnet_v2_2016_08_30.ckpt")
    parser.add_argument('--pretrained_model_path', type=str, default=None)

    # training steps configs
    parser.add_argument('--logging_every_n_steps', type=int, default=100)
    parser.add_argument('--save_every_n_steps', type=int, default=1000)
    parser.add_argument('--validation_every_n_steps', type=int, default=500)
    parser.add_argument('--summary_every_n_steps', type=int, default=100)

    # training learning rate configs
    parser.add_argument('--learning_rate_start', type=float, default=0.005)
    parser.add_argument('--learning_rate_decay_steps', type=int, default=10000)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.1)
    parser.add_argument('--learning_rate_staircase', type=bool, default=False)

    # training logs configs
    parser.add_argument('--logs_dir', type=str, default="./logs-densenet-121-test/", help='')
    parser.add_argument('--val_logs_dir', type=str, default="./logs-densenet-121-test/val/", help='')

    # test configs
    parser.add_argument('--trained_model', type=str,
                        default="./logs-inception-resnet-v2-reconstruct-ce-f1-aug-512/val/model.ckpt-41000", help='')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(_parse_arguments(sys.argv[1:]))
