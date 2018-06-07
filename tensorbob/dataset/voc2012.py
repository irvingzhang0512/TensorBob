import os
import numpy as np
from .dataset_utils import get_images_dataset_by_paths_config, \
    get_classification_labels_dataset_config, get_segmentation_labels_dataset_config
from .base_dataset import BaseDataset

__all__ = ['get_voc_classification_dataset', 'get_voc_segmentation_dataset']

# VOC2012_ROOT_DIR = "D:\\PycharmProjects\\data\\VOCdevkit\\VOC2012"
VOC2012_ROOT_DIR = "/home/ubuntu/data/VOC2012/train/VOC2012"
IMAGES_DIR = os.path.join(VOC2012_ROOT_DIR, 'JPEGImages')
CLASSIFICATION_CONFIG_DIR = os.path.join(VOC2012_ROOT_DIR, 'ImageSets', 'Main')
SEGMENTATION_CLASS_DIR = os.path.join(VOC2012_ROOT_DIR, 'SegmentationClass')

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor', 'background']
SEGMENTATION_COLOR_MAP = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                          [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                          [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                          [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                          [0, 192, 0], [128, 192, 0], [0, 64, 128], [0, 0, 0]]


def _get_classification_images_and_labels(mode):
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
    :param mode: 指定模式，train val trainval 三选一
    :param batch_size:
    :param kwargs:
    :return:
    """
    image_paths, label_paths = _get_classification_images_and_labels(mode)
    images_config = get_images_dataset_by_paths_config(image_paths, **kwargs)
    labels_config = get_classification_labels_dataset_config(label_paths)
    dataset_configs = [images_config, labels_config]
    train_mode = (mode == 'train')
    return BaseDataset(dataset_configs, batch_size, repeat=train_mode, shuffle=train_mode)


def _get_segmentation_images_and_labels(mode, train_size):
    label_file_names = os.listdir(SEGMENTATION_CLASS_DIR)
    image_paths = []
    label_paths = []
    for label_file_name in label_file_names:
        image_file_name = label_file_name[:label_file_name.find('.')] + '.jpg'
        image_paths.append(os.path.join(IMAGES_DIR, image_file_name))
        label_paths.append(os.path.join(SEGMENTATION_CLASS_DIR, label_file_name))
    ids = np.arange(len(image_paths))
    if mode == 'train':
        ids = ids[:train_size]
    elif mode == 'val':
        ids = ids[train_size:]
    elif mode == 'trainval':
        pass
    else:
        raise ValueError('unknown mode {}'.format(mode))
    return np.array(image_paths)[ids], np.array(label_paths)[ids]


def get_voc_segmentation_dataset(mode='train',
                                 train_size=2000,
                                 batch_size=32,
                                 val_image_height=None, val_image_width=None,
                                 **kwargs):
    print(kwargs)
    color_to_int_list = np.zeros(256 ** 3)
    for i, cords in enumerate(SEGMENTATION_COLOR_MAP):
        color_to_int_list[(cords[0] * 256 + cords[1]) * 256 + cords[2]] = i
    image_paths, label_paths = _get_segmentation_images_and_labels(mode, train_size)
    images_config = get_images_dataset_by_paths_config(image_paths, **kwargs)
    labels_config = get_segmentation_labels_dataset_config(label_paths, color_to_int_list,
                                                           image_width=val_image_width,
                                                           image_height=val_image_height)
    dataset_configs = [images_config, labels_config]
    train_mode = (mode == 'train')
    return BaseDataset(dataset_configs, batch_size, repeat=train_mode, shuffle=train_mode)
