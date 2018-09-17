import tensorflow as tf
from tensorbob.utils.preprocessing import random_crop, random_distort_color
from tensorbob.dataset.dataset_utils import CropType
from tensorbob.dataset.base_dataset import BaseDataset


__all__ = ['get_segmentation_dataset_config',
           'get_segmentation_dataset', ]


def get_segmentation_dataset_config(kwargs):
    """
    kwargs = {
        # 公共属性
        'random_flip_horizontal_flag': False,
        'random_flip_vertical_flag': False,
        'crop_type': bob.data.CropType.no_crop,
        'image_width': None,
        'image_height': None,
        'crop_width': None,
        'crop_height': None,

        # 标签特有属性
        'labels_paths': [],
        'color_to_int_list': [],
        'label_in_channels': 3,

        # 输入特有属性
        'images_paths': [],
        'norm_fn_first': None,
        'norm_fn_end': None,
        'random_distort_color_flag': False,
        'distort_color_fast_mode_flag': False,
    }
    :param kwargs:
    :return:
    """
    return {
        # 公共属性
        'random_flip_horizontal_flag': kwargs.get('random_flip_horizontal_flag'),
        'random_flip_vertical_flag': kwargs.get('random_flip_vertical_flag'),
        'crop_type': kwargs.get('crop_type'),
        'image_width': kwargs.get('image_width'),
        'image_height': kwargs.get('image_height'),
        'crop_width': kwargs.get('crop_width'),
        'crop_height': kwargs.get('crop_height'),

        # 标签特有属性
        'labels_paths': kwargs.get('labels_paths'),
        'color_to_int_list': kwargs.get('color_to_int_list'),
        'label_in_channels': kwargs.get('label_in_channels'),

        # 输入特有属性
        'images_paths': kwargs.get('images_paths'),
        'norm_fn_first': kwargs.get('norm_fn_first'),
        'norm_fn_end': kwargs.get('norm_fn_end'),
        'random_distort_color_flag': kwargs.get('random_distort_color_flag'),
        'distort_color_fast_mode_flag': kwargs.get('distort_color_fast_mode_flag'),
    }


def get_segmentation_dataset(dataset_configs,
                             epochs=1,
                             batch_size=8,
                             shuffle_flag=False,
                             shuffle_buffer_size=100,
                             prefetch_buffer_size=100,
                             ):
    """
    获取数据分割数据集
    要求：图片与标签的尺寸相同
    :param dataset_configs:
    :param epochs:
    :param batch_size:
    :param shuffle_flag:
    :param shuffle_buffer_size:
    :param prefetch_buffer_size:
    :return:                        BaseDataset 实例
    """
    # 标签属性
    labels_paths = dataset_configs.get('labels_paths')
    color_to_int_list = dataset_configs.get('color_to_int_list')
    label_in_channels = dataset_configs.get('label_in_channels') or 3
    if label_in_channels == 3 and color_to_int_list is None:
        raise ValueError('color_to_int_list cannot be None when in_channels = 3.')

    # 图片属性
    images_paths = dataset_configs.get('images_paths')
    norm_fn_first = dataset_configs.get('norm_fn_first')
    norm_fn_end = dataset_configs.get('norm_fn_end')
    random_distort_color_flag = dataset_configs.get('random_distort_color_flag') or False
    distort_color_fast_mode_flag = dataset_configs.get('distort_color_fast_mode_flag') or False

    # 通用属性
    random_flip_horizontal_flag = dataset_configs.get('random_flip_horizontal_flag') or False
    random_flip_vertical_flag = dataset_configs.get('random_flip_vertical_flag') or False
    crop_type = dataset_configs.get('crop_type') or CropType.no_crop
    image_width = dataset_configs.get('image_width')
    image_height = dataset_configs.get('image_height')
    crop_width = dataset_configs.get('crop_width')
    crop_height = dataset_configs.get('crop_height')

    if labels_paths is None:
        raise ValueError("labels_path cannot be None.")
    if images_paths is None:
        raise ValueError("images_paths cannot be None.")
    if len(images_paths) != len(labels_paths):
        raise ValueError("images_paths and labels_paths must have same length.")
    if labels_paths is None:
        raise ValueError("labels_path cannot be None")

    def _cur_parse_image_fn(image_path, label_path):
        # 读取图片与标签
        # 思路：将图片与标签在通道axis合并，然后再进行各类操作
        cur_image = tf.image.decode_jpeg(tf.read_file(image_path), channels=3)
        cur_label = tf.image.decode_jpeg(tf.read_file(label_path), channels=label_in_channels)
        merged_image = tf.concat([cur_image, cur_label], axis=-1)

        # 通用操作 - 镜像
        if random_flip_horizontal_flag:
            merged_image = tf.image.random_flip_left_right(merged_image)
        if random_flip_vertical_flag:
            merged_image = tf.image.random_flip_up_down(merged_image)

        # 通用操作 - 切片
        if crop_type is CropType.no_crop:
            if image_width is not None and image_height is not None:
                merged_image = tf.image.resize_images(merged_image, [image_height, image_width],
                                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        elif crop_type is CropType.random_normal:
            if crop_width is None or crop_height is None:
                raise ValueError('crop_width and crop_height must not be None when using normal random crop')
            if image_width is None or image_height is None:
                raise ValueError('image_width and image_height must not be None when using normal random crop')
            merged_image = tf.image.resize_images(merged_image,
                                                  [image_height, image_width],
                                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            merged_image = random_crop(merged_image, crop_height, crop_width)
        else:
            raise ValueError('undown crop type {}'.format(crop_type))

        # 将混合图片切分为 输入和标签
        channels = tf.split(merged_image, 3 + label_in_channels, axis=2)
        cur_image = tf.concat(channels[:3], axis=-1)
        cur_label = tf.concat(channels[3:], axis=-1)
        cur_label = tf.squeeze(cur_label)
        cur_label = tf.cast(tf.round(cur_label), tf.int32)

        # 输入图片操作
        if norm_fn_first:
            cur_image = norm_fn_first(tf.cast(cur_image, tf.uint8))
        if random_distort_color_flag:
            cur_image = random_distort_color(cur_image, distort_color_fast_mode_flag)
        if norm_fn_end:
            cur_image = norm_fn_end(cur_image)

        # 标签操作
        if color_to_int_list is not None and label_in_channels == 3:
            channels = tf.split(cur_label, 3, axis=2)
            cur_label = (256 * channels[0] + channels[1]) * 256 + channels[2]
            cur_label = tf.cast(tf.gather(color_to_int_list, tf.cast(cur_label, tf.int32)), tf.int32)

        return cur_image, cur_label

    dataset = tf.data.Dataset.zip(
        (tf.data.Dataset.from_tensor_slices(images_paths), tf.data.Dataset.from_tensor_slices(labels_paths))
    ).map(_cur_parse_image_fn)

    return BaseDataset(
        dataset=dataset,
        dataset_size=len(images_paths),
        batch_size=batch_size,
        shuffle=shuffle_flag,
        shuffle_buffer_size=shuffle_buffer_size,
        repeat=epochs,
        prefetch_buffer_size=prefetch_buffer_size,
    )
