import os
import numpy as np
from .dataset_utils import get_images_dataset_by_paths_config, get_classification_labels_dataset_config
from .base_dataset import BaseDataset


__all__ = ['get_imagenet_classification_dataset']

DATA_PATH = "/home/tensorflow05/data/ILSVRC2012"
IMAGE_DIRS = {"train": "ILSVRC2012_img_train",
              "val": "ILSVRC2012_img_val",
              "test": ""}
LABEL_DIRS = {"train": "ILSVRC2012_bbox_train",
              "val": "ILSVRC2012_bbox_val",
              "test": "ILSVRC2012_bbox_test_dogs"}
WNIDS_FILE = "imagenet_lsvrc_2015_synsets.txt"
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
BROKEN_IMAGE_VAL = []


def _get_wnids(data_path):
    with open(os.path.join(data_path, WNIDS_FILE)) as f:
        wnids = f.readlines()
    return [wnid.replace('\n', '') for wnid in wnids]


def _get_images_paths_and_labels(mode, data_path, labels_offset=0):
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
            if image in BROKEN_IMAGE_VAL:
                continue
            paths.append(os.path.join(data_path, IMAGE_DIRS[mode], image))
            labels.append(label)
    else:
        raise ValueError('unknown mode {}'.format(mode))
    return paths, labels


def get_imagenet_classification_dataset(mode,
                                        batch_size,
                                        data_path=DATA_PATH,
                                        shuffle_buffer_size=10000,
                                        prefetch_buffer_size=10000,
                                        labels_offset=0,
                                        **kwargs):
    paths, labels = _get_images_paths_and_labels(mode, data_path, labels_offset)
    images_config = get_images_dataset_by_paths_config(paths, **kwargs)
    labels_config = get_classification_labels_dataset_config(labels)
    dataset_config = [images_config, labels_config]
    train_mode = True if mode == 'train' else False
    return BaseDataset(dataset_config,
                       batch_size=batch_size,
                       shuffle=train_mode, shuffle_buffer_size=shuffle_buffer_size,
                       repeat=train_mode,
                       prefetch_buffer_size=prefetch_buffer_size)
