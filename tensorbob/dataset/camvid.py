import os
import numpy as np
import pandas as pd
from tensorbob.dataset.dataset_utils import get_images_dataset_by_paths_config, get_segmentation_labels_dataset_config
from tensorbob.dataset.base_dataset import BaseDataset, MergedDataset
from tensorflow.python.platform import tf_logging as logging


__all__ = ['get_camvid_segmentation_dataset', 'get_camvid_segmentation_merged_dataset']

DATA_PATH = "E:\\PycharmProjects\\data\\CamVid"
CLASS_DICT_FILE_NAME = "class_dict.csv"
MODES = ['train', 'val', 'test']


def _get_color_to_int_list(data_path):
    file_path = os.path.join(data_path, CLASS_DICT_FILE_NAME)
    df = pd.read_csv(file_path)
    color_to_int_list = np.zeros(256 ** 3)
    i = 0
    for cur_index in df.index:
        cur_rgb = df.loc[cur_index].values[1:]
        color_to_int_list[(cur_rgb[0] * 256 + cur_rgb[1]) * 256 + cur_rgb[2]] = i
        i += 1
    return color_to_int_list


def get_camvid_segmentation_dataset(mode='train',
                                    data_path=DATA_PATH,
                                    batch_size=32,
                                    shuffle_buffer_size=10000,
                                    prefetch_buffer_size=10000,
                                    repeat=1,
                                    label_image_height=None, label_image_width=None,
                                    **image_configs
                                    ):
    if mode not in MODES:
        raise ValueError('unknown mode {}, must be train, val or test.'.format())
    if label_image_height is None or label_image_width is None:
        raise ValueError('label_image_height and label_image_width must not be None')

    # 获取image_paths 和 labels_path
    image_dir = os.path.join(data_path, mode)
    label_dir = os.path.join(data_path, mode + '_labels')
    image_names = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, image_name) for image_name in image_names]
    label_paths = [os.path.join(label_dir, image_name[:image_name.find('.')] + '_L.png') for image_name in image_names]

    # 获取标签颜色与编号之间的关系
    color_to_int_list = _get_color_to_int_list(data_path)

    # 获取创建数据集所需参数
    images_config = get_images_dataset_by_paths_config(image_paths, **image_configs)
    labels_config = get_segmentation_labels_dataset_config(label_paths, color_to_int_list,
                                                           image_height=label_image_height,
                                                           image_width=label_image_width, )
    dataset_configs = [images_config, labels_config]

    # 创建数据集
    train_mode = (mode == 'train')
    logging.debug('successfully getting CamVid segmentation dataset for {} set'.format(mode))
    return BaseDataset(dataset_configs,
                       batch_size,
                       repeat=repeat,
                       shuffle=train_mode,
                       shuffle_buffer_size=shuffle_buffer_size,
                       prefetch_buffer_size=prefetch_buffer_size)


def get_camvid_segmentation_merged_dataset(train_args,
                                           val_args,
                                           label_image_height,
                                           label_image_width,
                                           data_path=DATA_PATH,
                                           batch_size=32,
                                           shuffle_buffer_size=10000,
                                           prefetch_buffer_size=10000,
                                           repeat=10,):
    if label_image_width is None or label_image_height is None:
        raise ValueError('label_image_height and label_image_width cannot be None')
    training_set = get_camvid_segmentation_dataset(data_path=data_path,
                                                   mode='train',
                                                   batch_size=batch_size,
                                                   repeat=repeat,
                                                   label_image_height=label_image_height,
                                                   label_image_width=label_image_width,
                                                   shuffle_buffer_size=shuffle_buffer_size,
                                                   prefetch_buffer_size=prefetch_buffer_size,
                                                   **train_args)
    val_set = get_camvid_segmentation_dataset(data_path=data_path,
                                              mode='val',
                                              batch_size=batch_size,
                                              repeat=1,
                                              label_image_height=label_image_height,
                                              label_image_width=label_image_width,
                                              prefetch_buffer_size=prefetch_buffer_size,
                                              **val_args)
    logging.debug('successfully getting segmentation merged dataset')
    return MergedDataset(training_set, val_set)
