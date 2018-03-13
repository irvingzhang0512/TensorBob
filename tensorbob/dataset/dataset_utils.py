import tensorflow as tf

from .preprocessing import *


def _get_images_by_path_dataset(dataset_config, batch_size):
    """
    {
        'type': 0,
        'src': [],
        'image_width': 500,
        'image_height': 500,
        'norm_fn': None,
        'crop_width': 300,
        'crop_height': 200,
        'central_crop_flag': True,
        'random_flip_horizontal_flag': True,
        'random_flip_vertical_flag': True,
    }
    :param dataset_config: 
    :param batch_size: 
    :return: 
    """
    file_paths = dataset_config['src']  # 原始数据
    image_width = dataset_config['image_width']  # 图片宽度
    image_height = dataset_config['image_height']  # 图片高度
    norm_fn = dataset_config['norm_fn']  # 归一化函数
    crop_width = dataset_config['crop_width']  # 切片宽
    crop_height = dataset_config['crop_height']  # 切片高
    central_crop_flag = dataset_config['central_crop_flag']  # 是否使用中心切片（默认随机切片）
    random_flip_horizontal_flag = dataset_config['random_flip_horizontal_flag']  # 随机水平镜像
    random_flip_vertical_flag = dataset_config['random_flip_vertical_flag']  # 随机垂直镜像

    def _cur_parse_image_fn(image_path):
        img_file = tf.read_file(image_path)
        cur_image = tf.image.decode_jpeg(img_file)
        if image_width is not None and image_height is not None:
            cur_image = tf.image.resize_images(cur_image, [image_height, image_width])
        if random_flip_horizontal_flag:
            cur_image = random_flip_horizontal(cur_image)
        if random_flip_vertical_flag:
            cur_image = random_flip_vertical(cur_image)
        return cur_image

    def _cur_preprocess_fn(images):
        if norm_fn:
            images = norm_fn(images)
        if crop_width is not None and crop_height is not None:
            if central_crop_flag:
                # 通过配置，可以使用中心切片
                images = central_crop(images, crop_height, crop_width)
            else:
                # 默认使用随机切片
                images = random_crop(images, crop_height, crop_width)
        return images

    cur_dataset = tf.data.Dataset.from_tensor_slices(file_paths) \
        .map(_cur_parse_image_fn) \
        .batch(batch_size=batch_size) \
        .map(_cur_preprocess_fn)
    return cur_dataset


def _get_labels_dataset(dataset_config, batch_size):
    """
    {
        'type': 1,
        'src': []
    }
    :param dataset_config: 
    :param batch_size: 
    :return: 
    """
    src = dataset_config['src']
    return tf.data.Dataset.from_tensor_slices(tf.constant(src)).batch(batch_size)


def get_dataset_by_config(dataset_config, batch_size):
    """
    通过config获取dataset以及dataset_size
    :param dataset_config: 
    :param batch_size: 
    :return: dataset 和 dataset_size
    """

    dataset_type = dataset_config['type']

    if dataset_type == 0:
        # 通过 file_path 获取 image
        return _get_images_by_path_dataset(dataset_config, batch_size), len(dataset_config['src'])
    elif dataset_type == 1:
        return _get_labels_dataset(dataset_config, batch_size), len(dataset_config['src'])
    else:
        raise ValueError('unknown dataset type {}'.format(dataset_type))


def get_image_by_path_dataset_config(file_paths, **kwargs):
    """
    通过file_paths获取图片
    :param file_paths: 
    :param kwargs:  image_width 
                    image_height 
                    norm_fn crop_width 
                    crop_height 
                    central_crop_flag 
                    random_flip_horizontal_flag 
                    random_flip_vertical_flag
    :return: 
    """
    dataset_config = {
        'type': 0,
        'src': file_paths,
        'image_width': kwargs.get('image_width') or 224,
        'image_height': kwargs.get('image_height') or 224,
        'norm_fn': kwargs.get('norm_fn'),
        'crop_width': kwargs.get('crop_width') or 224,
        'crop_height': kwargs.get('crop_height') or 224,
        'central_crop_flag': kwargs.get('central_crop_flag') or True,
        'random_flip_horizontal_flag': kwargs.get('random_flip_horizontal_flag') or True,
        'random_flip_vertical_flag': kwargs.get('random_flip_vertical_flag') or True,
    }
    return dataset_config


def get_labels_dataset_config(labels):
    """
    直接返回 labels
    :param labels: 
    :return: 
    """
    dataset_config = {
        'type': 1,
        'src': labels
    }
    return dataset_config
