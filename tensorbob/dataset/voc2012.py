import os
import numpy as np
from .dataset_utils import get_images_dataset_by_paths_config, \
    get_classification_labels_dataset_config, get_segmentation_labels_dataset_config
from .base_dataset import BaseDataset

__all__ = ['get_voc_classification_dataset', 'get_voc_segmentation_dataset']

# VOC2012_ROOT_DIR = "/home/ubuntu/data/voc2012/train/voc2012"
DATA_PATH = "/home/tensorflow05/data/voc2012"
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
    :param mode:
    :return:
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

    return keys, values


def get_voc_classification_dataset(data_path=DATA_PATH, mode='train', batch_size=32, **kwargs):
    """
    获取voc classification的 dataset
    :param data_path:   VOC数据所在目录
    :param mode:        指定模式，train val trainval 三选一
    :param batch_size:  dataset的batch size
    :param kwargs:      数据增强参数
    :return:            BaseDataset 实例
    """
    image_paths, label_paths = _get_classification_images_and_labels(data_path, mode)
    images_config = get_images_dataset_by_paths_config(image_paths, **kwargs)
    labels_config = get_classification_labels_dataset_config(label_paths)
    dataset_configs = [images_config, labels_config]
    train_mode = (mode == 'train')
    return BaseDataset(dataset_configs, batch_size, repeat=train_mode, shuffle=train_mode)


def _get_segmentation_images_and_labels(data_path, mode, train_size):
    """
    获取图像分割原始数据的 path ，包括输入图片已经label
    :param data_path:
    :param mode:
    :param train_size:
    :return:
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
        ids = ids[:train_size]
    elif mode == 'val':
        ids = ids[train_size:]
    elif mode == 'trainval':
        pass
    else:
        raise ValueError('unknown mode {}'.format(mode))
    return np.array(image_paths)[ids], np.array(label_paths)[ids]


def get_voc_segmentation_dataset(data_path=DATA_PATH,
                                 mode='train',
                                 train_size=2000,
                                 batch_size=32,
                                 label_image_height=None, label_image_width=None,
                                 **kwargs):
    """
    获取voc segmentation的 dataset
    要求自己保证image和label的尺寸相同
    label的尺寸通过 label_image_height, label_image_width确定
    image的尺寸通过几种切片方式确定

    :param data_path:           VOC数据所在文件夹
    :param mode:                train val trainval三种模式，由于图像分割中没有区分数据集，所以自己划分train val
    :param train_size:          设置train set的大小，方便获取train val数据集
    :param batch_size:          dataset的batch size
    :param label_image_height:  label的高
    :param label_image_width:   label的宽
    :param kwargs:              其他图像增强相关参数
    :return:                    BaseDataset 实例
    """
    color_to_int_list = np.zeros(256 ** 3)
    for i, cords in enumerate(SEGMENTATION_COLOR_MAP):
        color_to_int_list[(cords[0] * 256 + cords[1]) * 256 + cords[2]] = i
    image_paths, label_paths = _get_segmentation_images_and_labels(data_path, mode, train_size)
    images_config = get_images_dataset_by_paths_config(image_paths, **kwargs)
    labels_config = get_segmentation_labels_dataset_config(label_paths, color_to_int_list,
                                                           image_height=label_image_height,
                                                           image_width=label_image_width,)
    dataset_configs = [images_config, labels_config]
    train_mode = (mode == 'train')
    return BaseDataset(dataset_configs, batch_size, repeat=train_mode, shuffle=train_mode)
