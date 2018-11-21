import tensorflow as tf
import pandas as pd
import os
import cv2
import numpy as np


DATA_ROOT_DIR = "/home/tensorflow05/data/kaggle/protein"
TRAIN_CSV_FILE_NAME = "train.csv"
SUBMISSION_CSV_FILE_NAME = "submission.csv"
TF_RECORD_DIR = "/home/tensorflow05/data/kaggle/protein"


def _get_serialized_example(base_file_path, label=None):
    r_image = cv2.imread(base_file_path + '_red.png', cv2.IMREAD_GRAYSCALE)
    g_image = cv2.imread(base_file_path + '_green.png', cv2.IMREAD_GRAYSCALE)
    b_image = cv2.imread(base_file_path + '_blue.png', cv2.IMREAD_GRAYSCALE)
    y_image = cv2.imread(base_file_path + '_yellow.png', cv2.IMREAD_GRAYSCALE)
    img = np.stack((r_image, g_image, b_image, y_image), axis=2)
    if label is not None:
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(int64_list=tf.train.Int64List(value=img.reshape(-1))),
            'image_size': tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape)),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode()])),
        }))
    else:
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(int64_list=tf.train.Int64List(value=img.reshape(-1))),
            'image_size': tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape)),
        }))
    return example.SerializeToString()


def create_tfrecords_files(mode='train'):
    """
    由于创建的文件太大，所以不使用
    :param mode:
    :return:
    """
    if mode not in ['train', 'test']:
        raise ValueError('Unknown mode {}'.format(mode))
    if mode == 'train':
        csv_file_path = os.path.join(DATA_ROOT_DIR, TRAIN_CSV_FILE_NAME)
        df = pd.read_csv(csv_file_path)
        image_names = df['Id']
        image_labels = df['Target']
    else:
        csv_file_path = os.path.join(DATA_ROOT_DIR, SUBMISSION_CSV_FILE_NAME)
        df = pd.read_csv(csv_file_path)
        image_names = df['Id']

    writer = None
    for cur_id, image_name in enumerate(image_names):
        if cur_id % 1000 == 0:
            if writer is not None:
                writer.flush()
                writer.close()
            writer = tf.python_io.TFRecordWriter(os.path.join(TF_RECORD_DIR,
                                                              '%s_%04d.tfrecord' % (mode, (cur_id / 1000) + 1)))
        cur_label = None
        if mode == 'train':
            cur_label = image_labels[cur_id]
        example_serialized = _get_serialized_example(os.path.join(DATA_ROOT_DIR, mode, image_name), cur_label)
        writer.write(example_serialized)
    writer.flush()
    writer.close()


if __name__ == '__main__':
    create_tfrecords_files('train')
    create_tfrecords_files('test')
