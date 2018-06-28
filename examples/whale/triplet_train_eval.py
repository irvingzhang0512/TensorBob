import tensorflow as tf
import tensorbob as bob
import numpy as np
import pandas as pd
import os
import argparse
import sys
from nets import nets_factory
from tensorflow.python.platform import tf_logging as logging
from sklearn.svm import SVC
import pickle

logging.set_verbosity(logging.DEBUG)


def main(args):
    # 构建计算图

    # 各种placeholder
    ph_image_paths = tf.placeholder(tf.string, name='image_paths')
    ph_image_labels = tf.placeholder(tf.int32, name='image_labels')
    ph_is_training = tf.placeholder(tf.bool, name='is_training')

    # 创建数据集
    dataset = _get_dataset(ph_image_paths, ph_image_labels, args)
    images_batch, labels_batch = dataset.next_batch

    # 搭建模型，并构建损失函数与train_op
    embeddings = _get_embeddings(images_batch, ph_is_training, args)
    embeddings = tf.nn.l2_normalize(embeddings, 1, 1e-10, name='embeddings')
    triplet_loss = _triplet_loss(tf.reshape(embeddings, (-1, 3, args.embedding_size)), args.alpha)
    train_op = _get_train_op(args)
    summary_op = tf.summary.merge_all()

    # fine-tune
    # init_fn = bob.variables.assign_from_checkpoint_fn(args.pre_trained_model_path,
    #                                                   bob.variables.get_variables_to_restore(include=args.var_include_list,
    #                                                                                          exclude=args.var_exclude_list),
    #                                                   ignore_missing_vars=True,
    #                                                   reshape_variables=False)

    labels_int_to_str, images_per_class, train_image_paths, number_of_images_per_class = _get_train_input_data(
        args.train_csv_file_path,
        args.train_images_dir)
    test_image_paths, raw_test_file_names = _get_test_input_data(args.test_csv_file_path, args.test_images_dir)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # init_fn(sess)
        saver.restore(sess, args.pre_trained_model_path)

        # 预测模式
        if args.mode != 'train':
            print('start evaluating...')
            _evaluate(args, sess,
                      dataset, ph_image_paths, ph_image_labels, ph_is_training,
                      embeddings, images_batch, labels_batch,
                      labels_int_to_str, images_per_class, train_image_paths, number_of_images_per_class,
                      test_image_paths, raw_test_file_names)
            return

        # 训练模式
        summary_writer = tf.summary.FileWriter(args.logs_dir,
                                               sess.graph)
        for i in range(args.epochs):
            # 获取实际训练数据
            triplet_samples = _get_triplet_samples(args, sess,
                                                   dataset, ph_image_paths, ph_image_labels, ph_is_training,
                                                   embeddings, images_batch, labels_batch,
                                                   labels_int_to_str, images_per_class, train_image_paths,
                                                   number_of_images_per_class, )
            triplet_samples = triplet_samples.reshape(-1)
            print(triplet_samples.shape)

            # 实际训练
            print('start training epoch %d...' % (i + 1))
            dataset.reset(sess, feed_dict={ph_image_paths: triplet_samples,
                                           ph_image_labels: np.arange(len(triplet_samples))})
            j = 0
            while True:
                try:
                    j += 1
                    if j % 10 == 0:
                        cur_loss, cur_triplet_loss, summary_string = sess.run([train_op, triplet_loss, summary_op],
                                                                              feed_dict={ph_is_training: True})
                        print('epoch %d, step %d, loss is %.4f, triplet_loss is %.4f' % (
                        i + 1, j, cur_loss, cur_triplet_loss))
                        summary_writer.add_summary(summary_string)
                    else:
                        sess.run(train_op, feed_dict={ph_is_training: True})
                except tf.errors.OutOfRangeError:
                    print('epoch %d training finished.' % (i + 1))
                    saver.save(sess, os.path.join(args.logs_dir, 'model.ckpt'), tf.train.get_or_create_global_step())
                    break


def _evaluate(args, sess,
              dataset, ph_image_paths, ph_image_labels, ph_is_training,
              embeddings_tensor, images_batch, labels_batch,
              labels_int_to_str, images_per_class, train_image_paths, number_of_images_per_class,
              test_image_paths, raw_test_file_names
              ):
    print('getting training embeddings...')
    train_labels = []
    for idx, number_of_images_for_one_class in enumerate(number_of_images_per_class):
        train_labels += [idx] * number_of_images_for_one_class
    assert len(train_labels) == len(train_image_paths)

    dataset.reset(sess,
                  feed_dict={ph_image_paths: train_image_paths, ph_image_labels: np.arange(len(train_image_paths))})
    train_embedding_array = np.zeros([len(train_image_paths), args.embedding_size])
    while True:
        try:
            cur_embeddings, cur_labels = sess.run([embeddings_tensor, labels_batch], feed_dict={
                ph_is_training: False
            })
            train_embedding_array[cur_labels, :] = cur_embeddings
        except tf.errors.OutOfRangeError:
            break

    print('getting test embeddings...')
    dataset.reset(sess, feed_dict={ph_image_paths: test_image_paths, ph_image_labels: np.arange(len(test_image_paths))})
    test_embedding_array = np.zeros([len(test_image_paths), args.embedding_size])
    while True:
        try:
            cur_embeddings, cur_labels = sess.run([embeddings_tensor, labels_batch], feed_dict={
                ph_is_training: False
            })
            test_embedding_array[cur_labels, :] = cur_embeddings
        except tf.errors.OutOfRangeError:
            break

    print('SVM train...')
    model = SVC(kernel='linear', probability=True)
    model.fit(train_embedding_array, train_labels)

    print('savint training model to local file.')
    with open('./target.pickle', 'wb') as outfile:
        pickle.dump((model, labels_int_to_str), outfile)

    print('SVM test...')
    predictions = model.predict_proba(test_embedding_array)
    predictions_indices = np.argsort(-predictions, axis=1)[:, :4]
    print('prediction results shape is', predictions_indices.shape)
    assert predictions_indices.shape[0] == len(raw_test_file_names)

    print('generating final csv file...')
    ids = []
    for i in range(predictions_indices.shape[0]):
        cur_prediction_indices = predictions_indices[i]
        res = 'new_whale'
        for cur_idx in cur_prediction_indices:
            res += ' ' + labels_int_to_str[cur_idx]
        ids.append(res)
    csv = pd.DataFrame({
        'Image': raw_test_file_names,
        'Id': ids
    })
    csv.to_csv('./res.csv', index=False, columns=['Image', 'Id'])


def _get_triplet_samples(args, sess,
                         dataset, ph_image_paths, ph_image_labels, ph_is_training,
                         embeddings_tensor, images_batch, labels_batch,
                         labels_int_to_str, images_per_class, image_paths, number_of_images_per_class, ):
    print('start getting triplet samples...')
    dataset.reset(sess, feed_dict={ph_image_paths: image_paths, ph_image_labels: np.arange(len(image_paths))})
    embedding_array = np.zeros([len(image_paths), args.embedding_size])
    print('start getting embeddings for every picture...')
    while True:
        try:
            cur_embeddings, cur_labels = sess.run([embeddings_tensor, labels_batch], feed_dict={
                ph_is_training: False
            })
            embedding_array[cur_labels, :] = cur_embeddings
        except tf.errors.OutOfRangeError:
            break
    print('start selecting triplets...')
    triplets = _select_triplets_image_paths(embedding_array, image_paths,
                                            len(images_per_class), number_of_images_per_class,
                                            args.alpha)
    return np.array(triplets)


def _select_triplets_image_paths(embeddings, image_paths, number_of_class, number_of_images_per_class, alpha):
    """
    使用numpy操作
    :param embeddings: shape为[-1, embedding_size]，代表每张图片的embedding
    :param image_paths: shape与 embedding 相同，代表每张图片的image_paths
    :param number_of_class:  一共有多少个class
    :param number_of_images_per_class: 每个class分别有多少图片
    :param alpha:
    :return: 返回list，其中每个元素shape为(3,)，分别代表 anchor, positive, negative 对应的image_path
    """
    embedding_array_start_idx = 0
    triplets = []

    for i in range(number_of_class):
        if i % 200 == 0:
            print('cur selecting no.%d class/%d.' % (i + 1, number_of_class))
            print('current triplets number is %d.' % len(triplets))
        # if len(triplets) > 1000:
        #     break
        for j in range(1, number_of_images_per_class[i]):
            anchor_idx = embedding_array_start_idx + j - 1
            neg_dists = np.sum(np.square(embeddings[anchor_idx] - embeddings), 1)
            for k in range(j, number_of_images_per_class[i]):
                positive_idx = embedding_array_start_idx + k
                pos_dist = np.sum(np.square(embeddings[anchor_idx] - embeddings[positive_idx]))
                neg_dists[embedding_array_start_idx:embedding_array_start_idx + number_of_images_per_class[i]] = np.NaN
                all_neg = np.where(np.logical_and(neg_dists - pos_dist < alpha, pos_dist < neg_dists))[0]
                number_of_negs = all_neg.shape[0]
                if number_of_negs > 0:
                    negative_idx = all_neg[np.random.randint(number_of_negs)]
                    triplets.append((image_paths[anchor_idx], image_paths[positive_idx], image_paths[negative_idx]))
        embedding_array_start_idx += number_of_images_per_class[i]
    return triplets


def _get_train_op(args):
    total_loss = tf.losses.get_total_loss()
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(args.learning_rate_start,
                                               global_step,
                                               args.learning_rate_decay_steps,
                                               args.learning_rate_decay_rate,
                                               args.learning_rate_staircase)

    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_op = bob.training.create_train_op(total_loss, optimizer, global_step)

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('total_loss', total_loss)

    return train_op


def _triplet_loss(embeddings, alpha):
    """
    :param embeddings: shape为(-1, 3, embedding_size)
    :param alpha:
    :return:
    """
    with tf.variable_scope('triplet_loss'):
        # embeddings = tf.nn.l2_normalize(embeddings, 1, 1e-10, name='embeddings')
        anchor, positive, negative = tf.unstack(embeddings, 3, 1)
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
    return loss


def _get_embeddings(images_batch, ph_is_training, args):
    model_fn = nets_factory.get_network_fn(args.model_name, args.embedding_size, args.weight_decay, ph_is_training)
    embeddings, _ = model_fn(images_batch,
                             global_pool=True,
                             dropout_keep_prob=args.dropout_keep_prob,
                             create_aux_logits=False,
                             )
    return embeddings


def _get_test_input_data(csv_file_path, images_dir):
    df = pd.read_csv(csv_file_path)
    raw_file_names = df['Image'].values
    image_paths = [os.path.join(images_dir, file_name) for file_name in raw_file_names]
    return image_paths, raw_file_names


def _get_train_input_data(csv_file_path, images_dir):
    labels_int_to_str = {}
    images_per_class = []
    image_paths = []
    number_of_images_per_class = []

    df = pd.read_csv(csv_file_path)
    cur_id = -1
    for c, group in df.groupby("Id"):
        if cur_id == -1:
            cur_id += 1
            continue
        labels_int_to_str[cur_id] = c
        images = group['Image'].values
        images = [os.path.join(images_dir, image) for image in images]

        image_paths += images
        number_of_images_per_class.append(len(images))
        images_per_class.append(images)
        cur_id += 1

    return labels_int_to_str, images_per_class, image_paths, number_of_images_per_class


def _get_dataset(ph_image_paths, ph_image_labels, args):
    labels_config = bob.data.get_classification_labels_dataset_config(ph_image_labels)
    image_config_dict = {
        'norm_fn_first': bob.data.norm_zero_to_one,
        'norm_fn_end': bob.data.norm_minus_one_to_one,
        'random_flip_horizontal_flag': True,
        'crop_type': bob.data.CropType.no_crop,
        'image_width': args.image_size,
        'image_height': args.image_size,
    }
    images_config = bob.data.get_images_dataset_by_paths_config(ph_image_paths, **image_config_dict)
    return bob.data.BaseDataset([images_config, labels_config], batch_size=args.batch_size)


def _parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="eval")

    parser.add_argument('--train_csv_file_path', type=str,
                        default="/home/tensorflow05/data/kaggle/humpback_whale_identification/train.csv")
    parser.add_argument('--train_images_dir', type=str,
                        default="/home/tensorflow05/data/kaggle/humpback_whale_identification/train")
    parser.add_argument('--test_csv_file_path', type=str,
                        default="/home/tensorflow05/data/kaggle/humpback_whale_identification/sample_submission.csv")
    parser.add_argument('--test_images_dir', type=str,
                        default="/home/tensorflow05/data/kaggle/humpback_whale_identification/test")
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    parser.add_argument('--dropout_keep_prob', type=float, default=0.8)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--embedding_size', type=int, default=1024)

    parser.add_argument('--learning_rate_start', type=float, default=0.0001)
    parser.add_argument('--learning_rate_decay_steps', type=int, default=10000)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)
    parser.add_argument('--learning_rate_staircase', type=bool, default=False)

    # parser.add_argument('--image_size', type=int, default=224)
    # parser.add_argument('--pre_trained_model_path', type=str,
    #                     default='/home/tensorflow05/data/pre-trained/slim/vgg_16.ckpt')
    # parser.add_argument('--var_include_list', type=list, default=['vgg_16'])
    # parser.add_argument('--var_exclude_list', type=list, default=['vgg_16/fc8'])
    # parser.add_argument('--model_name', type=str, default='vgg_16')
    parser.add_argument('--image_size', type=int, default=299)
    # parser.add_argument('--pre_trained_model_path', type=str,
    #                     default='/home/tensorflow05/data/pre-trained/slim/inception_v3.ckpt')
    parser.add_argument('--pre_trained_model_path', type=str,
                        default='./logs/model.ckpt-5583')
    parser.add_argument('--var_include_list', type=list, default=['InceptionV3'])
    parser.add_argument('--var_exclude_list', type=list, default=['InceptionV3/Logits'])
    parser.add_argument('--model_name', type=str, default='inception_v3')

    parser.add_argument('--logs_dir', type=str, default="./logs/", help='')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(_parse_arguments(sys.argv[1:]))
