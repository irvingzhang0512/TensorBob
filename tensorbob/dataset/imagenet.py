import os
import numpy as np
from tensorbob.dataset.dataset_utils import get_images_dataset_by_paths_config, get_classification_labels_dataset_config
from tensorbob.dataset.base_dataset import BaseDataset, MergedDataset
from tensorflow.python.platform import tf_logging as logging

__all__ = ['get_imagenet_classification_dataset', 'get_imagenet_classification_merged_dataset']

DATA_PATH = "/home/tensorflow05/data/ILSVRC2012"
IMAGE_DIRS = {"train": "ILSVRC2012_img_train",
              "val": "ILSVRC2012_img_val",
              "test": ""}
WNIDS_FILE_NAME = "imagenet_lsvrc_2015_synsets.txt"
VAL_LABEL_FILE_NAME = "imagenet_2012_validation_synset_labels.txt"
BROKEN_IMAGES_TRAIN = ['n02667093_4388.JPEG',
                       'n09246464_51105.JPEG',
                       'n04501370_2480.JPEG',
                       'n04501370_19125.JPEG',
                       'n04501370_16347.JPE',
                       'n04501370_3775.JPEG',
                       'n02690373_15966.JPEG',
                       'n02494079_12155.JPEG',
                       'n04522168_538.JPEG',
                       'n02514041_1625.JPEG',
                       'n02514041_15019.JPEG',
                       'n07714571_1691.JPEG',
                       'n02640242_29608.JPEG',
                       'n04505470_7109.JPEG',
                       'n04505470_5018.JPEG',
                       'n09256479_9451.JPEG',
                       'n01770393_6999.JPEG',
                       'n09256479_1094.JPEG',
                       'n09256479_108.JPEG',
                       ]


def _get_wnids(data_path):
    with open(os.path.join(data_path, WNIDS_FILE_NAME)) as f:
        wnids = f.readlines()
    return [wnid.replace('\n', '') for wnid in wnids]


def _get_images_paths_and_labels(mode, data_path, labels_offset):
    """
    获取imagenet中train/val数据集的所有图片路径已经对应的label
    :param mode:            选择是train还是val
    :param data_path:       保存imagenetde lujing
    :param labels_offset:   label编号的其实数字，默认为0
    :return:                imagenet中train/val数据集的所有图片路径，以及对应的label
    """
    wnids = _get_wnids(data_path)
    paths = []
    labels = []
    if mode == 'train':
        for i, wnid in enumerate(wnids):
            images = os.listdir(os.path.join(data_path, IMAGE_DIRS[mode], wnid))
            for image in images:
                if image in BROKEN_IMAGES_TRAIN:
                    continue
                paths.append(os.path.join(data_path, IMAGE_DIRS[mode], wnid, image))
                labels.append(i + labels_offset)
        ids = np.arange(0, len(labels))
        np.random.shuffle(ids)
        paths = np.array(paths)[ids]
        labels = np.array(labels)[ids]
    elif mode == 'val':
        label_str_to_num = {}
        for i, wnid in enumerate(wnids):
            label_str_to_num[wnid] = i + labels_offset
        with open(os.path.join(data_path, VAL_LABEL_FILE_NAME)) as f:
            ground_truths = f.readlines()
        ground_truths = [label_str_to_num[label.strip()] for label in ground_truths]
        images = sorted(os.listdir(os.path.join(data_path, IMAGE_DIRS[mode])))
        for image, label in zip(images, ground_truths):
            paths.append(os.path.join(data_path, IMAGE_DIRS[mode], image))
            labels.append(label)
    else:
        raise ValueError('unknown mode {}'.format(mode))
    logging.debug('successfully getting {} paths and labels'.format(mode))
    return paths, labels


def get_imagenet_classification_dataset(mode,
                                        batch_size,
                                        data_path=DATA_PATH,
                                        repeat=1,
                                        shuffle_buffer_size=10000,
                                        prefetch_buffer_size=10000,
                                        labels_offset=0,
                                        **kwargs):
    """
    根据条件获取 BaseDataset 对象
    请注意：图像分类名称与编号，通过 magenet_lsvrc_2015_synsets.txt 来确认
    需要先下载该文件，并添加到 ImageNet2012 根目录下
    下载地址：https://github.com/tensorflow/models/blob/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt
    :param repeat:
    :param mode:                    模式，train/val 二选一
    :param batch_size:              Batch Size
    :param data_path:               ImageNet2012 根目录
    :param shuffle_buffer_size:     buffer的大小
    :param prefetch_buffer_size:    prefetch的大小
    :param labels_offset:           分类标签从几开始计算
    :param kwargs:                  图像增强参数，具体可以参考 dataset_utils 中 get_images_dataset_by_paths_config 函数
    :return:                        BaseDataset 对象
    """
    paths, labels = _get_images_paths_and_labels(mode, data_path, labels_offset)
    images_config = get_images_dataset_by_paths_config(paths, **kwargs)
    labels_config = get_classification_labels_dataset_config(labels)
    dataset_config = [images_config, labels_config]
    train_mode = True if mode == 'train' else False
    return BaseDataset(dataset_configs=dataset_config,
                       batch_size=batch_size,
                       shuffle=train_mode,
                       shuffle_buffer_size=shuffle_buffer_size,
                       repeat=repeat,
                       prefetch_buffer_size=prefetch_buffer_size)


def get_imagenet_classification_merged_dataset(train_args,
                                               val_args,
                                               data_path=DATA_PATH,
                                               batch_size=32,
                                               repeat=10,
                                               shuffle_buffer_size=10000,
                                               prefetch_buffer_size=10000,
                                               labels_offset=0):
    """
    根据条件获取 MergedDataset 对象
    请注意：图像分类名称与编号，通过 magenet_lsvrc_2015_synsets.txt 来确认
    需要先下载该文件，并添加到 ImageNet2012 根目录下
    下载地址：https://github.com/tensorflow/models/blob/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt
    :param train_args:              训练集参数
    :param val_args:                验证集参数
    :param data_path:               ImageNet路径
    :param batch_size:              batch size
    :param repeat:                  训练集epoch数
    :param shuffle_buffer_size:     训练集shuffle参数
    :param prefetch_buffer_size:    训练集/验证集参数
    :param labels_offset:           label offset
    :return:                        MergedDataset
    """
    training_dataset = get_imagenet_classification_dataset('train',
                                                           batch_size=batch_size,
                                                           data_path=data_path,
                                                           shuffle_buffer_size=shuffle_buffer_size,
                                                           prefetch_buffer_size=prefetch_buffer_size,
                                                           labels_offset=labels_offset,
                                                           repeat=repeat,
                                                           **train_args)
    val_dataset = get_imagenet_classification_dataset('val',
                                                      batch_size=batch_size,
                                                      data_path=data_path,
                                                      prefetch_buffer_size=prefetch_buffer_size,
                                                      labels_offset=labels_offset,
                                                      **val_args)
    return MergedDataset(training_dataset, val_dataset)
