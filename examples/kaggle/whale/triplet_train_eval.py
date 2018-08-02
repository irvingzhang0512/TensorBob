import tensorflow as tf
import tensorbob as bob
import numpy as np
import pandas as pd
import os
import argparse
import sys
import pickle
import time
from nets import nets_factory
from tensorflow.python.platform import tf_logging as logging

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
    triplet_loss = _triplet_loss(tf.reshape(embeddings, (-1, 3, args.embedding_size)), args.alpha)
    global_step = tf.train.get_or_create_global_step()
    train_op = _get_train_op(args)
    summary_op = tf.summary.merge_all()

    # fine-tune model
    if args.fine_tune_model_path is not None:
        if args.model_name == 'nasnet_large':
            ckpt_reader = tf.train.load_checkpoint(args.fine_tune_model_path)
            variable_names = list(ckpt_reader.get_variable_to_shape_map().keys())
            var_names_to_values = {}
            for var in variable_names:
                if var.find('final_layer') != -1:
                    continue
                var_name = 'nasnet/' + var
                if not tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, var_name):
                    continue
                var_names_to_values[var_name] = ckpt_reader.get_tensor(var)
            init_fn = bob.variables.assign_from_values_fn(var_names_to_values)
            print('get to restore %d vars' % len(var_names_to_values))
        else:
            var_list = bob.variables.get_variables_to_restore(include=args.var_include_list,
                                                              exclude=args.var_exclude_list)
            init_fn = bob.variables.assign_from_checkpoint_fn(args.fine_tune_model_path,
                                                              var_list=var_list,
                                                              ignore_missing_vars=True,
                                                              reshape_variables=False)

    # 构建计算图结束，开始训练/预测

    # load local csv file
    labels_int_to_str, images_per_class, train_image_paths, train_image_labels, number_of_images_per_class = _get_train_input_data(
        args.train_csv_file_path,
        args.train_images_dir)
    test_image_paths, raw_test_file_names = _get_test_input_data(args.test_csv_file_path, args.test_images_dir)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # load fine-tune model
        if args.fine_tune_model_path is not None:
            init_fn(sess)

        # load pre-trained model
        if args.pre_trained_model_path is not None:
            saver.restore(sess, args.pre_trained_model_path)

        # 预测模式
        if args.mode != 'train':
            print('start evaluating...')
            if args.evaluation_algorithm not in ['SVM', 'NearestNeighbors']:
                raise ValueError('unknown evaluation algorithm {}'.format(args.evaluation_algorithm))
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
            if args.triplet_select_mode == 'random':
                triplet_samples = _get_random_triplet_samples(images_per_class, train_image_paths,
                                                              train_image_labels, number_of_images_per_class,
                                                              args.triplets_per_epoch)
            elif args.triplet_select_mode == 'hard':
                triplet_samples = _get_hard_triplet_samples(args, sess,
                                                            dataset, ph_image_paths, ph_image_labels, ph_is_training,
                                                            embeddings, images_batch, labels_batch,
                                                            labels_int_to_str, images_per_class, train_image_paths,
                                                            number_of_images_per_class, )
            else:
                raise ValueError('unknown triplet select mode {}'.format(args.triplet_select_mode))
            triplet_samples = triplet_samples.reshape(-1)
            print('triplets samples size(after reshape(-1)) is', triplet_samples.shape)

            # 实际训练
            print('start training epoch %d...' % (i + 1))
            dataset.reset(sess, feed_dict={ph_image_paths: triplet_samples,
                                           ph_image_labels: np.arange(len(triplet_samples))})
            j = 0
            while True:
                try:
                    j += 1
                    if j % 100 == 0:
                        cur_step, cur_loss, summary_string = sess.run([global_step, train_op, summary_op],
                                                                      feed_dict={ph_is_training: True})
                        print('epoch %d, step %d, global step %d, loss is %.4f' % (
                            i + 1, j, cur_step, cur_loss))
                        summary_writer.add_summary(summary_string, global_step=cur_step)
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
    time_training = time.time()
    train_labels = []
    for idx, number_of_images_for_one_class in enumerate(number_of_images_per_class):
        train_labels += [idx] * number_of_images_for_one_class
    train_image_paths = train_image_paths[:len(train_labels)]
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
    print('train embedding shape is', train_embedding_array.shape)

    time_testing = time.time()
    print('training vectors cost %d s' % (time_testing - time_training))
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
    print('test embedding shape is', test_embedding_array.shape)

    time_classify = time.time()
    print('testing vectors cost %d s' % (time_classify - time_testing))
    ids = []
    if args.evaluation_algorithm == 'SVM':
        ids = _get_evaluation_res_by_svm(train_embedding_array, test_embedding_array,
                                         labels_int_to_str, train_labels)
    elif args.evaluation_algorithm == 'NearestNeighbors':
        ids = _get_evaluation_res_by_nearest_neighbors(train_embedding_array, test_embedding_array,
                                                       labels_int_to_str, train_labels)

    from datetime import datetime
    file_name = "./res_{}_{}.csv".format(args.evaluation_algorithm.lower(),
                                         datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M"))
    csv = pd.DataFrame({
        'Image': raw_test_file_names,
        'Id': ids
    })
    csv.to_csv(file_name, index=False, columns=['Image', 'Id'])
    print('classify cost %d s' % (time.time() - time_classify))


def _get_evaluation_res_by_nearest_neighbors(train_embedding_array, test_embedding_array,
                                             labels_int_to_str, train_labels):
    from sklearn.neighbors import NearestNeighbors
    print('NearestNeighbors train...')
    model = NearestNeighbors(n_neighbors=6)
    model.fit(train_embedding_array)

    print('saving training model to local file.')
    with open('./target_nearest_neighbors.pickle', 'wb') as outfile:
        pickle.dump((model, labels_int_to_str), outfile)

    print('NearestNeighbors test...')
    ids = []

    # NearestNeighbors
    distances_test, neighbors_test = model.kneighbors(test_embedding_array)
    distances_test, neighbors_test = distances_test.tolist(), neighbors_test.tolist()
    for distance, neighbour_ in zip(distances_test, neighbors_test):
        sample_result = []
        sample_classes = []
        for d, n in zip(distance, neighbour_):
            cur_cls_id = train_labels[n]
            cur_cls_name = labels_int_to_str[cur_cls_id]
            sample_classes.append(cur_cls_name)
            sample_result.append((cur_cls_name, d))

        if "new_whale" not in sample_classes:
            sample_result.append(("new_whale", 0))
        sample_result.sort(key=lambda x: x[1])
        sample_result = sample_result[:5]
        ids.append(" ".join([x[0] for x in sample_result]))

    return ids


def _get_evaluation_res_by_svm(train_embedding_array, test_embedding_array,
                               labels_int_to_str, train_labels):
    from sklearn.svm import SVC
    print('SVM train...')
    model = SVC(kernel='linear', probability=True)
    model.fit(train_embedding_array, train_labels)

    print('savint training model to local file.')
    with open('./target_svm.pickle', 'wb') as outfile:
        pickle.dump((model, labels_int_to_str), outfile)

    print('SVM test...')
    ids = []

    # SVM
    predictions = model.predict_proba(test_embedding_array)
    print('test predictions shape is', predictions.shape)
    predictions_indices = np.argsort(-predictions, axis=1)[:, :4]
    print('prediction results shape is', predictions_indices.shape)

    print('generating final csv file strings...')
    for i in range(predictions_indices.shape[0]):
        cur_prediction_indices = predictions_indices[i]
        res = 'new_whale'
        for cur_idx in cur_prediction_indices:
            res += ' ' + labels_int_to_str[cur_idx]
        ids.append(res)

    return ids


def _get_random_triplet_samples(images_per_class, image_paths, image_labels,
                                number_of_images_per_class, max_number_of_triplets):
    print('start getting triplet samples...')
    triplets_samples = []
    number_of_all_images = len(image_paths)
    cls_ids = np.arange(len(images_per_class))
    np.random.shuffle(cls_ids)

    for i in range(len(images_per_class)):
        if len(triplets_samples) >= max_number_of_triplets:
            break

        cur_cls_id = cls_ids[i]
        number_of_images_this_class = number_of_images_per_class[cur_cls_id]
        for j in range(1, number_of_images_this_class):
            anchor_id = j - 1
            for k in range(j, number_of_images_this_class):
                positive_id = k
                neg_id = None
                while neg_id is None or image_labels[neg_id] == cur_cls_id:
                    neg_id = np.random.randint(number_of_all_images)
                triplets_samples.append([images_per_class[cur_cls_id][anchor_id],
                                         images_per_class[cur_cls_id][positive_id],
                                         image_paths[neg_id]
                                         ])
    print('finished getting triplet samples...')
    return np.array(triplets_samples)[:max_number_of_triplets]


def _get_hard_triplet_samples(args, sess,
                              dataset, ph_image_paths, ph_image_labels, ph_is_training,
                              embeddings_tensor, images_batch, labels_batch,
                              labels_int_to_str, images_per_class, image_paths, number_of_images_per_class, ):
    print('start getting hard triplet samples...')
    start_time = time.time()
    dataset.reset(sess, feed_dict={ph_image_paths: image_paths, ph_image_labels: np.arange(len(image_paths))})
    embedding_array = np.zeros([len(image_paths), args.embedding_size])
    while True:
        try:
            cur_embeddings, cur_labels = sess.run([embeddings_tensor, labels_batch], feed_dict={
                ph_is_training: False
            })
            embedding_array[cur_labels, :] = cur_embeddings
        except tf.errors.OutOfRangeError:
            break
    embedding_time = time.time()
    print('getting embeddings cost %d s' % (embedding_time - start_time))
    print('start selecting triplets...')
    triplets = _select_hard_triplet_image_paths(embedding_array, image_paths,
                                                len(images_per_class), number_of_images_per_class,
                                                args.alpha)
    print('selecting hard triplets cost %d s' % (time.time() - embedding_time))
    return np.array(triplets)


def _select_hard_triplet_image_paths(embeddings, image_paths, number_of_class, number_of_images_per_class,
                                     alpha):
    """
    使用numpy操作
    :param embeddings: shape为[-1, embedding_size]，代表每张图片的embedding
    :param image_paths: 代表每张图片的image_paths
    :param number_of_class:  一共有多少个class
    :param number_of_images_per_class: 每个class分别有多少图片
    :param alpha:
    :return: 返回list，其中每个元素shape为(3,)，分别代表 anchor, positive, negative 对应的image_path
    """
    embedding_array_start_idx = 0
    triplets = []
    ids = np.arange(number_of_class)
    # np.random.shuffle(ids)

    for i in range(number_of_class):
        if i % 200 == 0:
            print('cur selecting no.%d class/%d.' % (i + 1, number_of_class))
            print('current triplets number is %d.' % len(triplets))
        # if len(triplets) > 5000:
        #     break
        cur_class = ids[i]
        for j in range(1, number_of_images_per_class[cur_class]):
            anchor_idx = embedding_array_start_idx + j - 1
            neg_dists = np.sum(np.square(embeddings[anchor_idx] - embeddings), 1)
            for k in range(j, number_of_images_per_class[cur_class]):
                positive_idx = embedding_array_start_idx + k
                pos_dist = np.sum(np.square(embeddings[anchor_idx] - embeddings[positive_idx]))
                neg_dists[
                embedding_array_start_idx:embedding_array_start_idx + number_of_images_per_class[cur_class]] = np.NaN
                all_neg = np.where(neg_dists < pos_dist + alpha)[0]
                number_of_negs = all_neg.shape[0]
                if number_of_negs > 0:
                    negative_idx = all_neg[np.random.randint(number_of_negs)]
                    triplets.append((image_paths[anchor_idx], image_paths[positive_idx], image_paths[negative_idx]))
        embedding_array_start_idx += number_of_images_per_class[cur_class]
    return triplets


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
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
    return loss


def _get_embeddings(images_batch, ph_is_training, args):
    if args.model_name == 'nasnet_large':
        model_fn = nets_factory.get_network_fn(args.model_name, args.embedding_size, args.weight_decay, False)
        with tf.variable_scope('nasnet'):
            embeddings, _ = model_fn(images_batch)
            embeddings = tf.nn.l2_normalize(embeddings, axis=1, name='embeddings')
    else:
        model_fn = nets_factory.get_network_fn(args.model_name, args.embedding_size, args.weight_decay, ph_is_training)
        embeddings, _ = model_fn(images_batch,
                                 # global_pool=True,
                                 dropout_keep_prob=args.dropout_keep_prob,
                                 create_aux_logits=False,
                                 )
        embeddings = tf.nn.l2_normalize(embeddings, axis=1, name='embeddings')

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
    image_labels = []
    number_of_images_per_class = []

    df = pd.read_csv(csv_file_path)
    cur_id = -1
    new_whale_images = []
    for c, group in df.groupby("Id"):
        if cur_id == -1:
            new_whale_images = group['Image'].values
            new_whale_images = [os.path.join(images_dir, image) for image in new_whale_images]
            cur_id += 1
            continue
        labels_int_to_str[cur_id] = c
        images = group['Image'].values
        images = [os.path.join(images_dir, image) for image in images]

        image_paths += images
        image_labels += [cur_id] * len(images)
        number_of_images_per_class.append(len(images))
        images_per_class.append(images)
        cur_id += 1

    # add new whale images
    image_paths += new_whale_images
    image_labels += [-1] * len(new_whale_images)

    return labels_int_to_str, images_per_class, image_paths, image_labels, number_of_images_per_class


def _get_dataset(ph_image_paths, ph_image_labels, args):
    labels_config = bob.data.get_classification_labels_dataset_config(ph_image_labels)
    image_config_dict = {
        'norm_fn_first': bob.data.norm_zero_to_one,
        'norm_fn_end': bob.data.norm_minus_one_to_one,
        'crop_type': bob.data.CropType.no_crop,
        'image_width': args.image_size,
        'image_height': args.image_size,
    }
    if args.mode == 'train':
        image_config_dict['random_flip_horizontal_flag'] = True
        image_config_dict['random_distort_color_flag'] = True
    images_config = bob.data.get_images_dataset_by_paths_config(ph_image_paths, **image_config_dict)
    return bob.data.BaseDataset([images_config, labels_config], batch_size=args.batch_size)


def _parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--evaluation_algorithm', type=str, default="NearestNeighbors")
    parser.add_argument('--triplet_select_mode', type=str, default="hard")

    # local input file
    # parser.add_argument('--train_csv_file_path', type=str,
    #                     default="/home/tensorflow05/data/kaggle/humpback_whale_identification/train.csv")
    # parser.add_argument('--train_images_dir', type=str,
    #                     default="/home/tensorflow05/data/kaggle/humpback_whale_identification/train_crop")
    # parser.add_argument('--test_csv_file_path', type=str,
    #                     default="/home/tensorflow05/data/kaggle/humpback_whale_identification/sample_submission.csv")
    # parser.add_argument('--test_images_dir', type=str,
    #                     default="/home/tensorflow05/data/kaggle/humpback_whale_identification/test_crop")
    parser.add_argument('--train_csv_file_path', type=str,
                        default="/home/ubuntu/data/kaggle/humpback/train.csv")
    parser.add_argument('--train_images_dir', type=str,
                        default="/home/ubuntu/data/kaggle/humpback/train_crop")
    parser.add_argument('--test_csv_file_path', type=str,
                        default="/home/ubuntu/data/kaggle/humpback/sample_submission.csv")
    parser.add_argument('--test_images_dir', type=str,
                        default="/home/ubuntu/data/kaggle/humpback/test_crop")
    # parser.add_argument('--train_csv_file_path', type=str,
    #                     default="E:\\PycharmProjects\\data\kaggle\\humpback_whale_identification\\train.csv")
    # parser.add_argument('--train_images_dir', type=str,
    #                     default="E:\\PycharmProjects\\data\kaggle\\humpback_whale_identification\\train")
    # parser.add_argument('--test_csv_file_path', type=str,
    #                     default="E:\\PycharmProjects\\data\kaggle\\humpback_whale_identification\\sample_submission.csv")
    # parser.add_argument('--test_images_dir', type=str,
    #                     default="E:\\PycharmProjects\\data\kaggle\\humpback_whale_identification\\test")

    # training configs
    parser.add_argument('--batch_size', type=int, default=15)  # 必须是3的倍数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=.0)
    parser.add_argument('--dropout_keep_prob', type=float, default=0.8)
    parser.add_argument('--triplets_per_epoch', type=int, default=5000)

    # triplet configs
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--embedding_size', type=int, default=2048)

    # learning rate
    parser.add_argument('--learning_rate_start', type=float, default=0.00002)
    parser.add_argument('--learning_rate_decay_steps', type=int, default=10000)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)
    parser.add_argument('--learning_rate_staircase', type=bool, default=False)

    # model
    # vgg16
    # parser.add_argument('--model_name', type=str, default='vgg_16')
    # parser.add_argument('--image_size', type=int, default=224)
    # parser.add_argument('--pre_trained_model_path', type=str,
    #                     default='/home/tensorflow05/data/pre-trained/slim/vgg_16.ckpt')
    # parser.add_argument('--var_include_list', type=list, default=['vgg_16'])
    # parser.add_argument('--var_exclude_list', type=list, default=['vgg_16/fc8'])

    # inception v3
    # parser.add_argument('--model_name', type=str, default='inception_v3')
    # parser.add_argument('--image_size', type=int, default=299)
    # parser.add_argument('--var_include_list', type=list, default=['InceptionV3'])
    # parser.add_argument('--var_exclude_list', type=list, default=['InceptionV3/Logits'])
    # parser.add_argument('--fine_tune_model_path', type=str,
    #                     default='/home/tensorflow05/data/pre-trained/slim/inception_v3.ckpt')
    # parser.add_argument('--pre_trained_model_path', type=str,
    #                     default=None)

    # nasnet
    parser.add_argument('--model_name', type=str, default='nasnet_large')
    parser.add_argument('--image_size', type=int, default=331)
    # parser.add_argument('--fine_tune_model_path', type=str,
    #                     default="E:\\PycharmProjects\\data\\slim\\nasnet\\model.ckpt")
#     parser.add_argument('--fine_tune_model_path', type=str,
#                         default="/home/ubuntu/data/slim/nasnet/model.ckpt")
#     parser.add_argument('--pre_trained_model_path', type=str,
#                         default=None)

    # inception resnet v2
    # parser.add_argument('--image_size', type=int, default=299)
    # parser.add_argument('--var_include_list', type=list, default=['InceptionResnetV2'])
    # parser.add_argument('--var_exclude_list', type=list, default=['InceptionResnetV2/Logits'])
    # parser.add_argument('--model_name', type=str, default='inception_resnet_v2')
    # parser.add_argument('--fine_tune_model_path', type=str,
    #                     default='/home/tensorflow05/data/pre-trained/slim/inception_resnet_v2_2016_08_30.ckpt')
    # parser.add_argument('--pre_trained_model_path', type=str, default=None)

    parser.add_argument('--fine_tune_model_path', type=str,
                        default=None)
    parser.add_argument('--pre_trained_model_path', type=str,
                        default='./logs_nasnet_2048_hard_samples/model.ckpt-33732')

    # logs
    parser.add_argument('--logs_dir', type=str, default="./logs_nasnet_2048_hard_samples/", help='')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(_parse_arguments(sys.argv[1:]))
