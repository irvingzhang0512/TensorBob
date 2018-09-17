import os
import numpy as np
import pandas as pd
from tensorbob.dataset.segmentation_dataset_utils import get_segmentation_dataset
from tensorbob.dataset.base_dataset import MergedDataset
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
                                    shuffle_buffer_size=100,
                                    prefetch_buffer_size=100,
                                    repeat=1,
                                    **image_configs
                                    ):
    """
    image_configs 举例
    {
        # 公共属性
        'random_flip_horizontal_flag': False,
        'random_flip_vertical_flag': False,
        'crop_type': bob.data.CropType.no_crop,
        'image_width': None,
        'image_height': None,
        'crop_width': None,
        'crop_height': None,

        # 输入特有属性
        'norm_fn_first': None,
        'norm_fn_end': None,
        'random_distort_color_flag': False,
        'distort_color_fast_mode_flag': False,
    }
    :param mode:
    :param data_path:
    :param batch_size:
    :param shuffle_buffer_size:
    :param prefetch_buffer_size:
    :param repeat:
    :param image_configs:
    :return:
    """
    if mode not in MODES:
        raise ValueError('unknown mode {}, must be train, val or test.'.format())
    if image_configs is None:
        image_configs = {}

    # 获取image_paths 和 labels_path
    image_dir = os.path.join(data_path, mode)
    label_dir = os.path.join(data_path, mode + '_labels')
    image_names = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, image_name) for image_name in image_names]
    label_paths = [os.path.join(label_dir, image_name[:image_name.find('.')] + '_L.png') for image_name in image_names]

    image_configs['labels_paths'] = label_paths
    image_configs['images_paths'] = image_paths
    image_configs['color_to_int_list'] = _get_color_to_int_list(data_path)

    return get_segmentation_dataset(image_configs,
                                    epochs=repeat,
                                    batch_size=batch_size,
                                    shuffle_flag=(mode == 'train'),
                                    shuffle_buffer_size=shuffle_buffer_size,
                                    prefetch_buffer_size=prefetch_buffer_size,)


def get_camvid_segmentation_merged_dataset(train_args,
                                           val_args,
                                           data_path=DATA_PATH,
                                           batch_size=32,
                                           shuffle_buffer_size=10000,
                                           prefetch_buffer_size=10000,
                                           repeat=10,):
    training_set = get_camvid_segmentation_dataset(data_path=data_path,
                                                   mode='train',
                                                   batch_size=batch_size,
                                                   repeat=repeat,
                                                   shuffle_buffer_size=shuffle_buffer_size,
                                                   prefetch_buffer_size=prefetch_buffer_size,
                                                   **train_args)
    val_set = get_camvid_segmentation_dataset(data_path=data_path,
                                              mode='val',
                                              batch_size=batch_size,
                                              repeat=1,
                                              prefetch_buffer_size=prefetch_buffer_size,
                                              **val_args)
    logging.debug('successfully getting segmentation merged dataset')
    return MergedDataset(training_set, val_set)
