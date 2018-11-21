import tensorflow as tf
import tensorbob as bob
from nets import inception_resnet_v2
import pandas as pd
import numpy as np
import os
import sys
import argparse
import tensorflow.contrib.slim as slim
from tensorflow.python.framework.errors_impl import OutOfRangeError
from tensorflow.python.platform import tf_logging as logging

logging.set_verbosity(logging.DEBUG)


def create_dataset(args):
    with tf.variable_scope('preprocessing'):
        def _parse_rgby_images(base_file_name):
            r_img = tf.image.decode_png(tf.read_file(base_file_name + '_red.png'), channels=1)
            g_img = tf.image.decode_png(tf.read_file(base_file_name + '_green.png'), channels=1)
            b_img = tf.image.decode_png(tf.read_file(base_file_name + '_blue.png'), channels=1)
            y_img = tf.image.decode_png(tf.read_file(base_file_name + '_yellow.png'), channels=1)

            r_img += tf.cast((y_img / 2), tf.uint8)
            g_img += tf.cast((y_img / 2), tf.uint8)

            img = tf.concat((r_img, g_img, b_img), axis=2)

            img = tf.image.resize_images(img, (args.image_height, args.image_width))
            return tf.image.convert_image_dtype(img, tf.float32, name='input_images') * 2.0 - 1.0

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

            # train set
            label_dataset = tf.data.Dataset.from_tensor_slices(_get_label_ndarays(image_labels[:-args.val_size]))
            image_names = [os.path.join(args.data_root_path, args.mode, image_name)
                           for image_name in image_names[:-args.val_size]]
            image_dataset = tf.data.Dataset.from_tensor_slices(image_names).map(_parse_rgby_images)
            train_set = bob.dataset.BaseDataset(dataset=tf.data.Dataset.zip((image_dataset, label_dataset)),
                                                dataset_size=len(image_names) - args.val_size,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                shuffle_buffer_size=args.shuffle_buffer_size,
                                                repeat=args.epochs,
                                                )

            # val set
            label_dataset = tf.data.Dataset.from_tensor_slices(_get_label_ndarays(image_labels[-args.val_size:]))
            image_names = [os.path.join(args.data_root_path, args.mode, image_name) for image_name in
                           image_names[-args.val_size:]]
            image_dataset = tf.data.Dataset.from_tensor_slices(image_names).map(_parse_rgby_images)
            val_set = bob.dataset.BaseDataset(dataset=tf.data.Dataset.zip((image_dataset, label_dataset)),
                                              dataset_size=args.val_size,
                                              batch_size=args.batch_size, )

            return bob.dataset.MergedDataset(train_set, val_set)
        else:
            csv_file_path = os.path.join(args.data_root_path, args.submission_csv_file_name)
            df = pd.read_csv(csv_file_path)
            image_names = df['Id']
            image_names = [os.path.join(args.data_root_path, args.mode, image_name) for image_name in image_names]
            image_dataset = tf.data.Dataset.from_tensor_slices(image_names).map(_parse_rgby_images)
            file_name_dataset = tf.data.Dataset.from_tensor_slices(df['Id'])
            return bob.dataset.BaseDataset(dataset=tf.data.Dataset.zip((image_dataset, file_name_dataset)),
                                           dataset_size=len(image_names),
                                           batch_size=args.batch_size, )


def get_model(x, args, is_training=False, ):
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


def get_loss(logits, labels):
    tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
    return tf.losses.get_total_loss()


def fbeta_score_macro(y_true, y_pred, beta=1, threshold=0.1):
    # https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(tf.greater(tf.cast(y_pred, 'float32'), threshold), 'float32')

    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)

    f1 = (1 + beta ** 2) * p * r / ((beta ** 2) * p + r + 1e-7)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return tf.reduce_mean(f1)


def get_metrics(total_loss, logits, labels, args):
    f1 = fbeta_score_macro(labels, tf.sigmoid(logits), threshold=args.sigmoid_threshold)
    summary_loss, update_loss = tf.metrics.mean(total_loss, name='loss')
    summary_f1, update_f1 = tf.metrics.mean(f1, name='f1')

    for metric in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES):
        tf.add_to_collection('RESET_OPS',
                             tf.assign(metric, tf.zeros(metric.get_shape(), metric.dtype)))

    with tf.control_dependencies(tf.get_collection('RESET_OPS')):
        after_reset_loss = tf.identity(update_loss, 'logging_loss')
        after_reset_f1 = tf.identity(update_f1, 'logging_f1')

    tf.summary.scalar('loss', summary_loss)
    tf.summary.scalar('f1', summary_f1)

    return [summary_f1, summary_loss], [update_f1, update_loss], [after_reset_f1, after_reset_loss]


def main(args):
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise ValueError('unknown mode {}'.format(args.mode))


def test(args):
    # 构建数据集
    dataset = create_dataset(args)
    input_images, input_file_names = dataset.next_batch

    # 构建模型
    ph_is_training = tf.placeholder(tf.bool)
    logits, end_points = get_model(input_images, args, ph_is_training)
    predictions = tf.cast(tf.sigmoid(logits) >= args.sigmoid_threshold, tf.int32)

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
    logits, end_points = get_model(input_images, args, ph_is_training)

    # 构建train_op
    total_loss = get_loss(logits, input_labels)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate_start)
    train_op = bob.training.create_train_op(total_loss,
                                            optimizer,
                                            global_step=tf.train.get_or_create_global_step())

    # 构建metrics
    summary_metrics, update_metrics, after_reset_metrics = get_metrics(total_loss, logits, input_labels, args)
    summary_op = tf.summary.merge_all()

    # 构建hooks
    val_feed_dict = {ph_is_training: False}
    validation_hook = bob.training.ValidationDatasetEvaluationHook(dataset,
                                                                   evaluate_every_n_steps=args.validation_every_n_steps,

                                                                   metrics_reset_ops=tf.get_collection('RESET_OPS'),
                                                                   metrics_update_ops=update_metrics,
                                                                   evaluating_feed_dict=val_feed_dict,

                                                                   summary_op=summary_op,
                                                                   summary_writer=tf.summary.FileWriter(
                                                                       args.val_logs_dir,
                                                                       tf.get_default_graph()),

                                                                   saver_file_prefix=os.path.join(args.val_logs_dir,
                                                                                                  'model.ckpt'),
                                                                   )
    init_fn_hook = bob.training.InitFnHook(dataset.init)
    hooks = [validation_hook, init_fn_hook]

    # 其他
    def feed_fn():
        return {dataset.ph_handle: dataset.handle_strings[0],
                ph_is_training: True}

    # scaffold
    if args.pretrained_model_path is None:
        scaffold = None
    else:
        var_list = bob.variables.get_variables_to_restore(include=['InceptionResnetV2'],
                                                          exclude=['InceptionResnetV2/Logits'])
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
                       summary_op=summary_op,
                       summary_writer=tf.summary.FileWriter(args.logs_dir, tf.get_default_graph()),
                       save_every_n_steps=args.save_every_n_steps,
                       )


def _parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # base configs
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--num_classes', type=int, default=28)
    parser.add_argument('--sigmoid_threshold', type=float, default=0.5)
    parser.add_argument('--data_root_path', type=str,
                        default="/home/tensorflow05/data/kaggle/protein")
    parser.add_argument('--train_csv_file_name', type=str,
                        default="train.csv")
    parser.add_argument('--submission_csv_file_name', type=str,
                        default="sample_submission.csv")
    parser.add_argument('--image_height', type=int, default=299)
    parser.add_argument('--image_width', type=int, default=299)

    # training base configs
    parser.add_argument('--val_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=.00005)
    parser.add_argument('--dropout_keep_prob', type=float, default=0.8)
    parser.add_argument('--shuffle_buffer_size', type=int, default=1000)
    parser.add_argument('--pretrained_model_path', type=str,
                        default="/home/tensorflow05/data/pre-trained/slim/inception_resnet_v2_2016_08_30.ckpt")

    # training steps configs
    parser.add_argument('--logging_every_n_steps', type=int, default=100)
    parser.add_argument('--save_every_n_steps', type=int, default=4000)
    parser.add_argument('--validation_every_n_steps', type=int, default=2000)
    parser.add_argument('--summary_every_n_steps', type=int, default=100)

    # training learning rate configs
    parser.add_argument('--learning_rate_start', type=float, default=0.0001)
    parser.add_argument('--learning_rate_decay_steps', type=int, default=10000)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)
    parser.add_argument('--learning_rate_staircase', type=bool, default=False)

    # training logs configs
    parser.add_argument('--logs_dir', type=str, default="./logs/", help='')
    parser.add_argument('--val_logs_dir', type=str, default="./logs/val/", help='')

    # test configs
    parser.add_argument('--trained_model', type=str, default="./logs/val/model.ckpt-4000", help='')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(_parse_arguments(sys.argv[1:]))
