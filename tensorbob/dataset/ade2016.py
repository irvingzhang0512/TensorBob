import os
from tensorbob.dataset.dataset_utils import get_images_dataset_by_paths_config, \
    get_segmentation_labels_dataset_config
from tensorbob.dataset.base_dataset import BaseDataset, MergedDataset
from tensorflow.python.platform import tf_logging as logging

__all__ = ['get_ade_segmentation_dataset', 'get_ade_segmentation_merged_dataset']

ROOT_DIR = "/home/tensorflow05/data/ade/ADEChallengeData2016"
TRAINING_DIR_NAME = "training"
VAL_DIR_NAME = "validation"
TESTING_DIR_NAME = "testing"
TESTING_LIST_FILE_NAME = 'list.txt'


def _get_images_and_labels_path(data_path, mode):
    images_dir = os.path.join(data_path, 'images')
    annotations_dir = os.path.join(data_path, 'annotations')
    if mode == 'train':
        image_file_names = os.listdir(os.path.join(images_dir, TRAINING_DIR_NAME))
        file_paths = [os.path.join(images_dir, TRAINING_DIR_NAME, image_name) for image_name in image_file_names]
        labels_paths = [os.path.join(annotations_dir, TRAINING_DIR_NAME,
                                     image_name[:image_name.find('.')] + '.png') for image_name in image_file_names]
        return file_paths, labels_paths
    elif mode == 'val':
        image_file_names = os.listdir(os.path.join(images_dir, VAL_DIR_NAME))
        file_paths = [os.path.join(images_dir, VAL_DIR_NAME, image_name) for image_name in image_file_names]
        labels_paths = [os.path.join(annotations_dir, VAL_DIR_NAME,
                                     image_name[:image_name.find('.')] + '.png') for image_name in image_file_names]
        return file_paths, labels_paths
    elif mode == 'test':
        with open(os.path.join(data_path, TESTING_LIST_FILE_NAME)) as f:
            image_file_names = f.readlines()
        image_file_names = [image_file_name.replace('\n', '') for image_file_name in image_file_names]
        file_paths = [os.path.join(images_dir, TESTING_DIR_NAME, image_name) for image_name in image_file_names]
        return file_paths, None
    else:
        raise ValueError('unknown mode {}'.format(mode))


def get_ade_segmentation_dataset(data_path=ROOT_DIR,
                                 mode='train',
                                 batch_size=32,
                                 shuffle_buffer_size=10000,
                                 prefetch_buffer_size=10000,
                                 repeat=1,
                                 label_image_height=None, label_image_width=None,
                                 **kwargs):
    image_paths, label_paths = _get_images_and_labels_path(data_path, mode)
    images_config = get_images_dataset_by_paths_config(image_paths, **kwargs)
    dataset_configs = [images_config]
    if mode != 'test':
        labels_config = get_segmentation_labels_dataset_config(label_paths,
                                                               in_channels=1,
                                                               image_height=label_image_height,
                                                               image_width=label_image_width, )
        dataset_configs.append(labels_config)

    train_mode = (mode == 'train')
    logging.debug('successfully getting segmentation dataset for {} set'.format(mode))
    return BaseDataset(dataset_configs,
                       batch_size,
                       repeat=repeat,
                       shuffle=train_mode,
                       shuffle_buffer_size=shuffle_buffer_size,
                       prefetch_buffer_size=prefetch_buffer_size)


def get_ade_segmentation_merged_dataset(train_args,
                                        val_args,
                                        data_path=ROOT_DIR,
                                        batch_size=32,
                                        shuffle_buffer_size=10000,
                                        prefetch_buffer_size=10000,
                                        repeat=10,
                                        label_image_height=None, label_image_width=None):
    training_set = get_ade_segmentation_dataset(data_path=data_path,
                                                mode='train',
                                                batch_size=batch_size,
                                                repeat=repeat,
                                                label_image_height=label_image_height,
                                                label_image_width=label_image_width,
                                                shuffle_buffer_size=shuffle_buffer_size,
                                                prefetch_buffer_size=prefetch_buffer_size,
                                                **train_args)
    val_set = get_ade_segmentation_dataset(data_path=data_path,
                                           mode='val',
                                           batch_size=batch_size,
                                           repeat=1,
                                           label_image_height=label_image_height,
                                           label_image_width=label_image_width,
                                           prefetch_buffer_size=prefetch_buffer_size,
                                           **val_args)
    logging.debug('successfully getting segmentation merged dataset')
    return MergedDataset(training_set, val_set)
