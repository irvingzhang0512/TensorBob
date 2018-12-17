import tensorflow as tf
import tensorbob as bob
import pandas as pd
import numpy as np
import os
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from protein_utils import generate_k_folds, get_k_folds


def create_dataset(args, cur_folds_index=0):
    with tf.variable_scope('preprocessing'):
        def _parse_rgby_images(base_file_name):
            r_img = tf.image.decode_png(tf.read_file(base_file_name + '_red.png'), channels=1)
            g_img = tf.image.decode_png(tf.read_file(base_file_name + '_green.png'), channels=1)
            b_img = tf.image.decode_png(tf.read_file(base_file_name + '_blue.png'), channels=1)
            y_img = tf.image.decode_png(tf.read_file(base_file_name + '_yellow.png'), channels=1)
            img = tf.concat((r_img, g_img, b_img, y_img), axis=2)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize_images(img, (args.image_height, args.image_width))
            return img * 2.0 - 1.0

        def _image_augumentation(image):
            image = (image + 1.0) / 2.0
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.rot90(image, tf.random_uniform([], maxval=4, dtype=tf.int32))
            if args.model_in_channels == 3:
                channels = tf.split(axis=2, num_or_size_splits=4, value=image)
                raw_image = tf.concat(channels[:3], axis=2)
                raw_image = bob.utils.random_distort_color(raw_image, False, scope='distort_color')
                image = tf.concat([raw_image, channels[3]], axis=2)
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
            image_names = np.array(df['Id'])
            image_labels = _get_label_ndarays(df['Target'])

            if args.k_folds == 1:
                msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=args.val_percent, random_state=0)
                for train_index, val_index in msss.split(image_names, image_labels):
                    train_image_names = image_names[train_index]
                    train_image_labels = image_labels[train_index]
                    val_image_names = image_names[val_index]
                    val_image_labels = image_labels[val_index]
            else:
                if args.k_folds_generate:
                    generate_k_folds(args.k_folds, image_labels)
                train_index, val_index = get_k_folds(cur_folds_index)
                train_image_names = image_names[train_index]
                train_image_labels = image_labels[train_index]
                val_image_names = image_names[val_index]
                val_image_labels = image_labels[val_index]

            # train set
            train_label_dataset = tf.data.Dataset.from_tensor_slices(train_image_labels)
            train_image_names = [os.path.join(args.data_root_path, args.mode, image_name)
                                 for image_name in train_image_names]
            train_image_dataset = tf.data.Dataset.from_tensor_slices(train_image_names) \
                .map(_parse_rgby_images).map(_image_augumentation)
            train_set = bob.dataset.BaseDataset(dataset=tf.data.Dataset.zip((train_image_dataset, train_label_dataset)),
                                                dataset_size=len(train_image_labels),
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                shuffle_buffer_size=args.shuffle_buffer_size,
                                                repeat=args.epochs,
                                                )

            # val set
            val_label_dataset = tf.data.Dataset.from_tensor_slices(val_image_labels)
            val_image_names = [os.path.join(args.data_root_path, args.mode, image_name)
                               for image_name in val_image_names]
            val_image_dataset = tf.data.Dataset.from_tensor_slices(val_image_names).map(_parse_rgby_images)
            val_set = bob.dataset.BaseDataset(dataset=tf.data.Dataset.zip((val_image_dataset, val_label_dataset)),
                                              dataset_size=len(val_image_labels),
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
                                           batch_size=args.batch_size)
