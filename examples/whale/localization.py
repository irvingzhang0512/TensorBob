import tensorbob as bob
import tensorflow as tf
import os
import sys
import numpy as np
import pandas as pd
import argparse
import pickle
from nets import nets_factory
from lxml import etree
from tensorflow.python.platform import tf_logging as logging

logging.set_verbosity(logging.DEBUG)


def main(args):
    # 构建计算图

    # 各种placeholder
    ph_is_training = tf.placeholder(tf.bool, name='is_training')

    # 创建数据集
    if args.mode == 'train':
        image_paths, image_labels = _get_image_paths_and_labels(args)
    elif args.mode == 'eval':
        image_paths, image_labels = _get_eval_image_paths_and_labels(args.train_csv_file_path,
                                                                     args.train_images_dir)
    else:
        raise ValueError('unknown mode {}'.format(args.mode))

    dataset = _get_dataset(args, image_paths, image_labels)
    images_batch, labels_batch = dataset.next_batch

    # 搭建模型以及对应的train_op
    locations = _get_locations(images_batch, ph_is_training, args)

    if args.mode == 'train':
        train(args, dataset, labels_batch, locations, ph_is_training)
    else:
        evaluate(args, dataset, labels_batch, locations, ph_is_training)


def evaluate(args, dataset, labels_batch, locations, ph_is_training):
    print('start evaluating...')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # load pre-trained model
        if args.pre_trained_model_path is not None:
            saver.restore(sess, args.pre_trained_model_path)

        dataset.reset(sess)
        location_res = np.zeros([dataset.size, 4])
        while True:
            try:
                cur_locations, cur_labels = sess.run([locations, labels_batch], feed_dict={ph_is_training: True})
                location_res[cur_labels, :] = cur_locations
            except tf.errors.OutOfRangeError:
                with open('./locations_train.pickle', 'wb') as outfile:
                    location_res[location_res > 1] = 1
                    location_res[location_res < 0] = 0

                    pickle.dump(location_res, outfile)
                    print('dump successfully...')
                    print(location_res.shape)
                break


def _get_eval_image_paths_and_labels(csv_file_path, images_dir):
    df = pd.read_csv(csv_file_path)
    cur_id = -1
    paths = []
    for c, group in df.groupby("Id"):
        if cur_id == -1:
            cur_id += 1
            continue
        images = group['Image'].values
        images = [os.path.join(images_dir, image) for image in images]
        paths += images
    return np.array(paths), np.arange(len(paths))


def train(args, dataset, labels_batch, locations, ph_is_training):
    print('start training')

    _get_regression_loss(locations, labels_batch)
    train_op = _get_train_op(args)
    summary_op = tf.summary.merge_all()

    # fine-tune model
    if args.fine_tune_model_path is not None:
        var_list = bob.variables.get_variables_to_restore(include=args.var_include_list,
                                                          exclude=args.var_exclude_list)
        init_fn = bob.variables.assign_from_checkpoint_fn(args.fine_tune_model_path,
                                                          var_list=var_list,
                                                          ignore_missing_vars=True,
                                                          reshape_variables=False)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # load fine-tune model
        if args.fine_tune_model_path is not None:
            init_fn(sess)

        # load pre-trained model
        if args.pre_trained_model_path is not None:
            saver.restore(sess, args.pre_trained_model_path)

        # 训练模式
        summary_writer = tf.summary.FileWriter(args.logs_dir, sess.graph)

        for i in range(args.epochs):
            print('start training epoch %d...' % (i + 1))
            dataset.reset(sess)
            j = 0
            while True:
                try:
                    j += 1
                    if j % 4 == 0:
                        cur_loss, summary_string = sess.run([train_op, summary_op],
                                                            feed_dict={ph_is_training: True})
                        print('epoch %d, step %d, loss is %.4f' % (
                            i + 1, j, cur_loss))
                        summary_writer.add_summary(summary_string)
                    else:
                        sess.run(train_op, feed_dict={ph_is_training: True})
                except tf.errors.OutOfRangeError:
                    print('epoch %d training finished.' % (i + 1))
                    saver.save(sess, os.path.join(args.logs_dir, 'model.ckpt'), tf.train.get_or_create_global_step())
                    break


def _get_train_op(args):
    total_loss = tf.losses.get_total_loss()
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(args.learning_rate_start,
                                               global_step,
                                               args.learning_rate_decay_steps,
                                               args.learning_rate_decay_rate,
                                               args.learning_rate_staircase)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = bob.training.create_train_op(total_loss, optimizer, global_step)

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('total_loss', total_loss)

    return train_op


def _get_regression_loss(locations, labels):
    return tf.losses.mean_squared_error(labels, locations)


def _get_locations(images_batch, ph_is_training, args):
    model_fn = nets_factory.get_network_fn(args.model_name, 4, args.weight_decay, ph_is_training)
    locations, _ = model_fn(images_batch,
                            global_pool=True,
                            dropout_keep_prob=args.dropout_keep_prob,
                            create_aux_logits=False,
                            )
    return locations


def _get_dataset(args, image_paths, image_labels):
    labels_config = bob.data.get_classification_labels_dataset_config(image_labels)
    image_config_dict = {
        'norm_fn_first': bob.data.norm_zero_to_one,
        'norm_fn_end': bob.data.norm_minus_one_to_one,
        'crop_type': bob.data.CropType.no_crop,
        'image_width': args.image_size,
        'image_height': args.image_size,
    }
    images_config = bob.data.get_images_dataset_by_paths_config(image_paths, **image_config_dict)
    return bob.data.BaseDataset([images_config, labels_config], batch_size=args.batch_size,
                                shuffle=(args.mode == 'train'))


def _get_image_paths_and_labels(args):
    labels = []
    paths = []

    for file_name in os.listdir(args.annotations_dir):
        image_path = os.path.join(args.images_dir, file_name[:file_name.find('.')] + '.jpg')
        paths.append(image_path)

        xml_file_path = os.path.join(args.annotations_dir, file_name)
        with open(xml_file_path, 'r') as f:
            xml_str = f.read()
        xml_map = _recursive_parse_xml_to_dict(etree.fromstring(xml_str))['annotation']
        image_width = float(xml_map['size']['width'])
        image_height = float(xml_map['size']['height'])
        xmin = float(xml_map['object'][0]['bndbox']['xmin'])
        ymin = float(xml_map['object'][0]['bndbox']['ymin'])
        xmax = float(xml_map['object'][0]['bndbox']['xmax'])
        ymax = float(xml_map['object'][0]['bndbox']['ymax'])
        labels.append(
            [ymin / image_height, xmin / image_width, ymax / image_height, xmax / image_width])

    return np.array(paths), np.array(labels)


def _recursive_parse_xml_to_dict(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = _recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def _parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="eval")

    # local input file
    parser.add_argument('--images_dir', type=str,
                        default="/home/tensorflow05/data/kaggle/humpback_whale_identification/location/Images")
    parser.add_argument('--annotations_dir', type=str,
                        default="/home/tensorflow05/data/kaggle/humpback_whale_identification/location/Annotations")
    parser.add_argument('--train_csv_file_path', type=str,
                        default="/home/tensorflow05/data/kaggle/humpback_whale_identification/train.csv")
    parser.add_argument('--train_images_dir', type=str,
                        default="/home/tensorflow05/data/kaggle/humpback_whale_identification/train")
    parser.add_argument('--test_csv_file_path', type=str,
                        default="/home/tensorflow05/data/kaggle/humpback_whale_identification/sample_submission.csv")
    parser.add_argument('--test_images_dir', type=str,
                        default="/home/tensorflow05/data/kaggle/humpback_whale_identification/test")

    # training configs
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--weight_decay', type=float, default=.0)
    parser.add_argument('--dropout_keep_prob', type=float, default=0.8)

    # learning rate
    parser.add_argument('--learning_rate_start', type=float, default=0.00001)
    parser.add_argument('--learning_rate_decay_steps', type=int, default=10000)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)
    parser.add_argument('--learning_rate_staircase', type=bool, default=False)

    # model
    parser.add_argument('--image_size', type=int, default=299)
    # parser.add_argument('--fine_tune_model_path', type=str,
    #                     default='/home/tensorflow05/data/pre-trained/slim/inception_v3.ckpt')
    # parser.add_argument('--fine_tune_model_path', type=str,
    #                     default='/home/ubuntu/data/slim/inception_v3.ckpt')
    # parser.add_argument('--pre_trained_model_path', type=str,
    #                     default=None)

    parser.add_argument('--fine_tune_model_path', type=str,
                        default=None)
    parser.add_argument('--pre_trained_model_path', type=str,
                        default='./logs_localization/model.ckpt-896')
    parser.add_argument('--var_include_list', type=list, default=['InceptionV3'])
    parser.add_argument('--var_exclude_list', type=list, default=['InceptionV3/Logits'])
    parser.add_argument('--model_name', type=str, default='inception_v3')

    # logs
    parser.add_argument('--logs_dir', type=str, default="./logs_localization/")

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(_parse_arguments(sys.argv[1:]))
