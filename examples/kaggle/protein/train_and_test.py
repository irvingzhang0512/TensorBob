# coding=utf-8
import tensorflow as tf
import tensorbob as bob
from protein_model import get_densenet_model, get_resnet_v2_50_model, \
    get_inception_resnet_v2_model, get_inception_v3_model
import pandas as pd
import numpy as np
import os
import sys
import argparse
from tensorflow.python.framework.errors_impl import OutOfRangeError
from tensorflow.python.platform import tf_logging as logging
from protein_utils import focal_loss, focal_loss_v2, fbeta_score_macro, f1_loss
from protein_data import create_dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.set_verbosity(logging.DEBUG)

# # 0.3119
# thresholds = np.array([0.407, 0.441, 0.161, 0.145, 0.299, 0.129, 0.25, 0.414, 0.01, 0.028, 0.021, 0.125,
#                        0.113, 0.387, 0.602, 0.001, 0.137, 0.199, 0.176, 0.25, 0.095, 0.29, 0.159, 0.255,
#                        0.231, 0.363, 0.117, 0., ])

# 0.3094
thresholds = 0.2


# # 0.3094
# thresholds = 0.4


# # 0.3100
# thresholds = 0.15


def get_encoder_decoder_model(x, args, is_training):
    if args.model_in_channels == 3:
        channels = tf.split(axis=3, num_or_size_splits=4, value=x)
        x = tf.concat(axis=3, values=channels[:3])

    if args.model == 'inception_v3':
        return get_inception_v3_model(x, args, is_training, args.model_out_channels)
    elif args.model == 'inception_resnet_v2':
        return get_inception_resnet_v2_model(x, args, is_training, args.model_out_channels)
    elif args.model == 'resnet_v2_50':
        return get_resnet_v2_50_model(x, args, is_training, args.model_out_channels)
    elif args.model == 'densenet':
        return get_densenet_model(x, args, is_training, args.model_out_channels)
    else:
        raise ValueError('Unknown model {}'.format(args.model))


def get_classifier_and_reconstruction_loss(args,
                                           classifier_logits, classifier_labels,
                                           reconstruct_image, raw_images):
    # classifier loss
    if args.loss == 'ce':
        classifier_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=classifier_labels,
                                                          logits=classifier_logits,
                                                          weights=1, )
    elif args.loss == 'f1':
        classifier_loss = f1_loss(classifier_labels, tf.sigmoid(classifier_logits), thresholds)
    elif args.loss == 'focal':
        classifier_loss = focal_loss(classifier_logits, classifier_labels)
    else:
        raise ValueError('Unknown loss function {}'.format(args.loss))

    merged_list = [
        tf.summary.scalar(args.loss, classifier_loss),
        tf.summary.scalar('regularization_loss', tf.losses.get_regularization_loss()),
    ]

    # reconstruction loss
    if args.with_mse:
        if args.model_out_channels == 3:
            channels = tf.split(axis=3, num_or_size_splits=4, value=raw_images)
            raw_images = tf.concat(axis=3, values=channels[:3])
        elif args.model_out_channels == 1:
            channels = tf.split(axis=3, num_or_size_splits=4, value=raw_images)
            raw_images = channels[3]
        reconstruction_loss = tf.losses.mean_squared_error(predictions=reconstruct_image, labels=raw_images)
        merged_list.append(tf.summary.scalar('reconstruction_loss', reconstruction_loss))
        merged_list.append(tf.summary.image('raw_image', raw_images))
        merged_list.append(tf.summary.image('reconstgruct_image', reconstruct_image))
    seperate_loss_summary_op = tf.summary.merge(merged_list)

    return tf.losses.get_total_loss(), seperate_loss_summary_op


def get_metrics(total_loss, logits, labels):
    with tf.variable_scope('metrics_related'):
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


def evaluate(args):
    # 构建数据集
    dataset = create_dataset(args)
    input_images, input_labels = dataset.next_batch

    # 构建模型
    ph_is_training = tf.placeholder(tf.bool)
    reconstruct_image, logits, end_points = get_encoder_decoder_model(input_images, args,
                                                                      is_training=ph_is_training,)

    n_sigmoid = tf.nn.sigmoid(logits)
    pred = n_sigmoid > thresholds

    with tf.Session() as sess:
        # 初始化模型
        sess.run(tf.global_variables_initializer())
        sess.run(dataset.iterator.initializer)
        saver = tf.train.Saver()
        latest_model = tf.train.latest_checkpoint(os.path.join(args.logs_dir, 'val'))
        saver.restore(sess, latest_model)

        all_pred = None
        all_labels = None
        i = 0
        try:
            while True:
                cur_pred, cur_labels = sess.run([pred, input_labels], feed_dict={ph_is_training: False})
                all_pred = cur_pred if all_pred is None else np.concatenate((all_pred, cur_pred), axis=0)
                all_labels = cur_labels if all_labels is None else np.concatenate((all_labels, cur_labels), axis=0)
                i += 1
                if i % 100 == 0:
                    print(i, cur_pred)
        except OutOfRangeError:
            pass
        import sklearn
        final_f1 = sklearn.metrics.f1_score(all_labels, all_pred, average='macro')
    print('final f1: %.4f' % final_f1)


def test(args):
    # 构建数据集
    dataset = create_dataset(args)
    input_images, input_file_names = dataset.next_batch

    # 构建模型
    ph_is_training = tf.placeholder(tf.bool)
    reconstruct_image, logits, end_points = get_encoder_decoder_model(input_images, args,
                                                                      is_training=ph_is_training,)
    logits_sigmoids = tf.sigmoid(logits)
    predictions = tf.cast(logits_sigmoids >= thresholds, tf.int32)

    if args.k_folds == 1:
        df = pd.DataFrame(columns=('Id', 'Predicted'))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(dataset.iterator.initializer)
            saver = tf.train.Saver()
            latest_model = tf.train.latest_checkpoint(os.path.join(args.logs_dir, 'val'))
            saver.restore(sess, latest_model)

            try:
                while True:
                    cur_predictions, cur_file_names = sess.run([predictions, input_file_names],
                                                               feed_dict={ph_is_training: False})
                    for cur_predictions, cur_file_name in zip(cur_predictions, cur_file_names):
                        label_str = " ".join([str(cur_num) for cur_num in np.where(cur_predictions == 1)[0]])
                        df = df.append({'Id': cur_file_name.decode(), 'Predicted': label_str}, True)
            except OutOfRangeError:
                pass
        from datetime import datetime
        file_name = "./{}_{}.csv".format(args.model, datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M"))
    else:
        total_sigmoids = None
        for i in range(args.k_folds):
            cur_fold_sigmoids = []
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(dataset.iterator.initializer)
                saver = tf.train.Saver()
                latest_model = tf.train.latest_checkpoint(os.path.join(args.logs_dir, str(i), 'val'))
                saver.restore(sess, latest_model)
                try:
                    while True:
                        cur_sigmoids = sess.run(logits_sigmoids, feed_dict={ph_is_training: False})
                        cur_fold_sigmoids.append(cur_sigmoids)
                except OutOfRangeError:
                    pass
            cur_fold_sigmoids = np.concatenate(cur_fold_sigmoids, axis=0)
            total_sigmoids = cur_fold_sigmoids if total_sigmoids is None else total_sigmoids + cur_fold_sigmoids
        avg_sigmoids = total_sigmoids / args.k_folds

        df = pd.read_csv(os.path.join(args.data_root_path, args.submission_csv_file_name))
        submissions = []
        for cur_row in avg_sigmoids:
            cur_submission_list = np.nonzero(cur_row > thresholds)[0]
            if len(cur_submission_list) == 0:
                cur_submission_str = ''
            else:
                cur_submission_str = ' '.join(list([str(i) for i in cur_submission_list]))
            submissions.append(cur_submission_str)
        df['Predicted'] = submissions

        from datetime import datetime
        file_name = "./{}_ensemble_{}_folds_{}.csv".format(args.model, args.k_folds,
                                                           datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M"))

    if args.add_leak_data:
        leak_data_df = pd.read_csv(args.leak_data_file_path)
        leak_data_df.drop(['Extra', 'SimR', 'SimG', 'SimB', 'Target_noisey'], axis=1, inplace=True)
        leak_data_df.columns = ['Id', 'Leak']
        leak_data_df = leak_data_df.set_index('Id')
        df = df.set_index('Id')
        for cur_index in leak_data_df.index:
            if cur_index in df.index:
                df.loc[cur_index].Predicted = leak_data_df.loc[cur_index].Leak
        df.to_csv(file_name)
    else:
        df.to_csv(file_name, index=False)


def train_one_model(args, cur_folds_index):
    tf.reset_default_graph()
    cur_logs_dir = args.logs_dir
    if args.k_folds > 1:
        cur_logs_dir = os.path.join(cur_logs_dir, str(cur_folds_index))

    # 构建数据集
    dataset = create_dataset(args)
    input_images, input_labels = dataset.next_batch

    # 构建模型
    ph_is_training = tf.placeholder(tf.bool)
    reconstruct_image, logits, end_points = get_encoder_decoder_model(input_images, args,
                                                                      is_training=ph_is_training,)

    # 训练相关
    total_loss, seperate_loss_summary = get_classifier_and_reconstruction_loss(args,
                                                                               logits, input_labels,
                                                                               reconstruct_image, input_images,
                                                                               )
    lr = bob.training.learning_rate_exponential_decay(args.learning_rate_start,
                                                      tf.train.get_or_create_global_step(),
                                                      args.learning_rate_decay_steps,
                                                      args.learning_rate_decay_rate,
                                                      args.learning_rate_staircase)

    # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
    train_op = bob.training.create_train_op(total_loss,
                                            optimizer,
                                            global_step=tf.train.get_or_create_global_step())

    # 构建metrics
    summary_metrics, update_metrics, after_reset_metrics, metrics_summary_op = get_metrics(total_loss,
                                                                                           logits,
                                                                                           input_labels)
    training_summary_writer = tf.summary.FileWriter(cur_logs_dir, tf.get_default_graph())

    # 构建hooks
    summary_hook = bob.training.SummarySaverHook(summary_op=seperate_loss_summary,
                                                 save_steps=args.summary_every_n_steps,
                                                 summary_writer=training_summary_writer, )
    val_feed_dict = {ph_is_training: False}
    val_logs_dir = os.path.join(cur_logs_dir, 'val')

    validation_hook = bob.training.ValidationDatasetEvaluationHook(dataset,
                                                                   evaluate_every_n_steps=args.validation_every_n_steps,

                                                                   metrics_reset_ops=tf.get_collection('RESET_OPS'),
                                                                   metrics_update_ops=update_metrics,
                                                                   evaluating_feed_dict=val_feed_dict,

                                                                   summary_op=metrics_summary_op,
                                                                   summary_writer=tf.summary.FileWriter(
                                                                       val_logs_dir,
                                                                       tf.get_default_graph()),

                                                                   saver_file_prefix=os.path.join(val_logs_dir,
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
    bob.training.train(train_op, cur_logs_dir,
                       session_config=config,
                       hooks=hooks,
                       scaffold=scaffold,
                       logging_tensors=update_metrics,
                       logging_every_n_steps=args.logging_every_n_steps,
                       feed_fn=feed_fn,
                       summary_every_n_steps=args.summary_every_n_steps,
                       summary_op=metrics_summary_op,
                       summary_writer=training_summary_writer,
                       save_every_n_steps=args.save_every_n_steps,
                       )


def train(args):
    if args.k_folds == 1:
        train_one_model(args, None)
    else:
        for i in range(args.k_folds):
            train_one_model(args, i)


def main(args):
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    else:
        raise ValueError('unknown mode {}'.format(args.mode))


def _parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # base configs
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--num_classes', type=int, default=28)
    parser.add_argument('--data_root_path', type=str, default="/ssd/zhangyiyang/protein")
    parser.add_argument('--train_csv_file_name', type=str, default="train.csv")
    parser.add_argument('--submission_csv_file_name', type=str, default="sample_submission.csv")
    parser.add_argument('--image_height', type=int, default=512)
    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--model_in_channels', type=int, default=3)
    parser.add_argument('--model_out_channels', type=int, default=1)

    # k folds
    parser.add_argument('--k_folds', type=int, default=6)
    parser.add_argument('--k_folds_index', type=int, default=0)
    parser.add_argument('--k_folds_generate', type=bool, default=False)

    # training base configs
    parser.add_argument('--model', type=str, default='inception_v3')
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--with_mse', type=bool, default=False)
    parser.add_argument('--val_percent', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=.00001)
    parser.add_argument('--dropout_keep_prob', type=float, default=0.8)
    parser.add_argument('--shuffle_buffer_size', type=int, default=1000)
    parser.add_argument('--pretrained_model_includes', type=list, default=['InceptionV3'])
    parser.add_argument('--pretrained_model_excludes', type=list, default=None)
    parser.add_argument('--pretrained_model_path', type=str,
                        default='/ssd/zhangyiyang/slim/inception_v3.ckpt')

    # training steps configs
    parser.add_argument('--logging_every_n_steps', type=int, default=100)
    parser.add_argument('--save_every_n_steps', type=int, default=1800)
    parser.add_argument('--validation_every_n_steps', type=int, default=1800)
    parser.add_argument('--summary_every_n_steps', type=int, default=100)

    # training learning rate configs
    parser.add_argument('--learning_rate_start', type=float, default=0.03)
    parser.add_argument('--learning_rate_decay_steps', type=int, default=18000)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.1)
    parser.add_argument('--learning_rate_staircase', type=bool, default=True)

    # training logs configs
    parser.add_argument('--logs_dir', type=str, default="./logs-inception-v3-ce-ensemble/", help='')

    # test configs
    parser.add_argument('--add_leak_data', type=bool, default=True, help='')
    parser.add_argument('--leak_data_file_path', type=str, default="/ssd/zhangyiyang/protein/leak_data.csv", help='')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(_parse_arguments(sys.argv[1:]))
