# 数据集中文介绍：

import os
from .dataset_utils import get_images_path_dataset_config, get_classification_labels_dataset_config
from .base_dataset import BaseDataset

# VOC2012_ROOT_DIR = "D:\\PycharmProjects\\data\\VOCdevkit\\VOC2012"
VOC2012_ROOT_DIR = "/home/ubuntu/data/VOC2012/train/VOC2012"
IMAGES_DIR = os.path.join(VOC2012_ROOT_DIR, 'JPEGImages')
CLASSIFICATION_CONFIG_DIR = os.path.join(VOC2012_ROOT_DIR, 'ImageSets', 'Main')

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor', 'background']
SEGMENTATION_COLOR_MAP = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                          [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                          [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                          [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                          [0, 192, 0], [128, 192, 0], [0, 64, 128], [0, 0, 0]]


def _get_classification_paths_and_labels(mode='trainval'):
    """
    根据 mode 获取对应的 file_paths 和 labels
    :param mode:
    :return:
    """
    if mode not in ['train', 'val', 'trainval']:
        raise ValueError('Unknown mode: {}'.format(mode))

    result_dict = {}
    for i, class_name in enumerate(CLASSES):
        if i == 20:
            # background 不用处理
            break
        file_name = class_name + "_" + mode + '.txt'
        for line in open(os.path.join(CLASSIFICATION_CONFIG_DIR, file_name), 'r'):
            line = line.replace('  ', ' ').replace('\n', '')
            parts = line.split(' ')
            if int(parts[1]) == 1:
                result_dict[os.path.join(IMAGES_DIR, parts[0] + '.jpg')] = i

    keys = []
    values = []
    for key, value in result_dict.items():
        keys.append(key)
        values.append(value)

    return keys, values


def get_voc_classification_dataset(mode='train', batch_size=32, **kwargs):
    """
    获取voc classification的 dataset
    :param epochs: 重复次数
    :param mode: 指定模式，train val trainval 三选一
    :param batch_size:
    :param kwargs:
    :return:
    """
    paths, labels = _get_classification_paths_and_labels(mode)
    images_config = get_images_path_dataset_config(paths, **kwargs)
    labels_config = get_classification_labels_dataset_config(labels)
    dataset_configs = [images_config, labels_config]
    train_mode = (mode == 'train')
    return BaseDataset(dataset_configs, batch_size, repeat=train_mode, shuffle=train_mode)
