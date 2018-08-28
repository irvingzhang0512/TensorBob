import os
import numpy as np
from tensorbob.dataset.dataset_utils import get_images_dataset_by_paths_config, \
    get_classification_labels_dataset_config, get_segmentation_labels_dataset_config
from tensorbob.dataset.base_dataset import BaseDataset, MergedDataset
from tensorflow.python.platform import tf_logging as logging

__all__ = ['get_voc_classification_dataset', 'get_voc_classification_merged_dataset',
           'get_voc_segmentation_dataset', 'get_voc_segmentation_merged_dataset']

DATA_PATH = "/home/tensorflow05/data/VOCdevkit/VOC2012"
IMAGES_DIR_NAME = 'JPEGImages'
CLASSIFICATION_CONFIG_DIR_NAME = os.path.join('ImageSets', 'Main')
SEGMENTATION_CLASS_DIR_NAME = 'SegmentationClass'

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor', 'background']
SEGMENTATION_COLOR_MAP = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                          [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                          [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                          [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                          [0, 192, 0], [128, 192, 0], [0, 64, 128], [0, 0, 0]]


def _get_classification_images_and_labels(data_path, mode):
    """
    根据 mode 获取对应的 file_paths 和 labels
    """
    images_dir = os.path.join(data_path, IMAGES_DIR_NAME)
    classification_config_dir = os.path.join(data_path, CLASSIFICATION_CONFIG_DIR_NAME)

    if mode not in ['train', 'val', 'trainval']:
        raise ValueError('Unknown mode: {}'.format(mode))

    result_dict = {}
    for i, class_name in enumerate(CLASSES):
        if i == 20:
            # background 不用处理
            break
        file_name = class_name + "_" + mode + '.txt'
        for line in open(os.path.join(classification_config_dir, file_name), 'r'):
            line = line.replace('  ', ' ').replace('\n', '')
            parts = line.split(' ')
            if int(parts[1]) == 1:
                result_dict[os.path.join(images_dir, parts[0] + '.jpg')] = i

    keys = []
    values = []
    for key, value in result_dict.items():
        keys.append(key)
        values.append(value)

    logging.debug('successfully getting classification paths and labels for {} set'.format(mode))
    return keys, values


def get_voc_classification_dataset(data_path=DATA_PATH, mode='train', batch_size=32, repeat=1, **kwargs):
    """
    获取voc classification的 dataset
    :param data_path:   VOC数据所在目录
    :param mode:        指定模式，train val trainval 三选一
    :param batch_size:  dataset的batch size
    :param kwargs:      数据增强参数
    :param repeat:      epoch数量
    :return:            BaseDataset 实例
    """
    image_paths, label_paths = _get_classification_images_and_labels(data_path, mode)
    images_config = get_images_dataset_by_paths_config(image_paths, **kwargs)
    labels_config = get_classification_labels_dataset_config(label_paths)
    dataset_configs = [images_config, labels_config]
    train_mode = (mode == 'train')
    logging.debug('successfully getting classification dataset for {} set'.format(mode))
    return BaseDataset(dataset_configs, batch_size, repeat=repeat, shuffle=train_mode)


def get_voc_classification_merged_dataset(train_args,
                                          val_args,
                                          data_path=DATA_PATH,
                                          batch_size=32,
                                          repeat=10):
    training_set = get_voc_classification_dataset(data_path=data_path,
                                                  batch_size=batch_size,
                                                  mode='train',
                                                  repeat=repeat,
                                                  **train_args)
    val_set = get_voc_classification_dataset(data_path=data_path,
                                             batch_size=batch_size,
                                             mode='val',
                                             repeat=1,
                                             **val_args)
    logging.debug('successfully getting classification merged dataset')
    return MergedDataset(training_set, val_set)


def _get_segmentation_images_and_labels(data_path, mode, val_set_size):
    """
    获取图像分割原始数据和对应标签的 path
    """
    image_dir = os.path.join(data_path, IMAGES_DIR_NAME)
    segmentation_class_dir = os.path.join(data_path, SEGMENTATION_CLASS_DIR_NAME)

    label_file_names = os.listdir(segmentation_class_dir)
    image_paths = []
    label_paths = []
    for label_file_name in label_file_names:
        image_file_name = label_file_name[:label_file_name.find('.')] + '.jpg'
        image_paths.append(os.path.join(image_dir, image_file_name))
        label_paths.append(os.path.join(segmentation_class_dir, label_file_name))
    ids = np.arange(len(image_paths))
    if mode == 'train':
        ids = ids[:-val_set_size]
    elif mode == 'val':
        ids = ids[-val_set_size:]
    elif mode == 'trainval':
        pass
    else:
        raise ValueError('unknown mode {}'.format(mode))
    logging.debug('successfully getting segmentation paths and labels for {} set'.format(mode))
    return np.array(image_paths)[ids], np.array(label_paths)[ids]


def get_voc_segmentation_dataset(data_path=DATA_PATH,
                                 mode='train',
                                 val_set_size=2000,
                                 batch_size=32,
                                 shuffle_buffer_size=10000,
                                 prefetch_buffer_size=10000,
                                 repeat=1,
                                 label_image_height=None, label_image_width=None,
                                 **kwargs):
    """
    获取voc segmentation的 dataset
    要求自己保证image和label的尺寸相同
    label的尺寸通过 label_image_height, label_image_width确定
    image的尺寸通过几种切片方式确定

    :param data_path:           VOC数据所在文件夹
    :param mode:                train val trainval三种模式，由于图像分割中没有区分数据集，所以自己划分train val
    :param val_set_size:        设置验证集尺寸
    :param batch_size:          batch size
    :param prefetch_buffer_size:
    :param shuffle_buffer_size:
    :param repeat:              epoch数量
    :param label_image_height:  label的高
    :param label_image_width:   label的宽
    :param kwargs:              其他图像增强相关参数
    :return:                    BaseDataset 实例
    """
    color_to_int_list = np.zeros(256 ** 3)
    for i, cords in enumerate(SEGMENTATION_COLOR_MAP):
        color_to_int_list[(cords[0] * 256 + cords[1]) * 256 + cords[2]] = i
    image_paths, label_paths = _get_segmentation_images_and_labels(data_path, mode, val_set_size)
    images_config = get_images_dataset_by_paths_config(image_paths, **kwargs)
    labels_config = get_segmentation_labels_dataset_config(label_paths, color_to_int_list,
                                                           image_height=label_image_height,
                                                           image_width=label_image_width, )
    dataset_configs = [images_config, labels_config]
    train_mode = (mode == 'train')

    logging.debug('successfully getting segmentation dataset for {} set'.format(mode))
    return BaseDataset(dataset_configs,
                       batch_size,
                       repeat=repeat,
                       shuffle=train_mode,
                       shuffle_buffer_size=shuffle_buffer_size,
                       prefetch_buffer_size=prefetch_buffer_size)


def get_voc_segmentation_merged_dataset(train_args,
                                        val_args,
                                        data_path=DATA_PATH,
                                        val_set_size=2000,
                                        batch_size=32,
                                        shuffle_buffer_size=10000,
                                        prefetch_buffer_size=10000,
                                        repeat=10,
                                        label_image_height=None, label_image_width=None):
    """
    获取voc segmentation的 merged dataset
    要求自己保证image和label的尺寸相同
    label的尺寸通过 label_image_height, label_image_width确定
    image的尺寸通过几种切片方式确定
    :param train_args:              训练集参数
    :param val_args:                验证集参数
    :param data_path:               voc数据路径
    :param val_set_size:            验证集数量
    :param batch_size:              batch size
    :param shuffle_buffer_size:     训练集参数
    :param prefetch_buffer_size:    训练集/验证集参数
    :param repeat:                  训练集参数
    :param label_image_height:      标签尺寸
    :param label_image_width:       标签尺寸
    :return:                        MergedDataset
    """
    training_set = get_voc_segmentation_dataset(data_path=data_path,
                                                mode='train',
                                                val_set_size=val_set_size,
                                                batch_size=batch_size,
                                                repeat=repeat,
                                                label_image_height=label_image_height,
                                                label_image_width=label_image_width,
                                                shuffle_buffer_size=shuffle_buffer_size,
                                                prefetch_buffer_size=prefetch_buffer_size,
                                                **train_args)
    val_set = get_voc_segmentation_dataset(data_path=data_path,
                                           mode='val',
                                           val_set_size=val_set_size,
                                           batch_size=batch_size,
                                           repeat=1,
                                           label_image_height=label_image_height,
                                           label_image_width=label_image_width,
                                           prefetch_buffer_size=prefetch_buffer_size,
                                           **val_args)
    logging.debug('successfully getting segmentation merged dataset')
    return MergedDataset(training_set, val_set)
