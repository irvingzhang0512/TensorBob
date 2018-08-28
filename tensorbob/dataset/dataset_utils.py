import tensorflow as tf
from enum import Enum
from tensorbob.utils.preprocessing import central_crop, random_crop, \
    random_crop_inception, random_crop_vgg, random_distort_color


__all__ = ['get_dataset_by_config',
           'get_images_dataset_by_paths_config',
           'get_classification_labels_dataset_config',
           'get_segmentation_labels_dataset_config',
           'CropType',
           ]


def get_dataset_by_config(dataset_config):
    """
    通过config获取dataset以及dataset_size
    目前支持以下几种：
    0: image path
    1: classification labels
    :param dataset_config: 字典，具体参考下面代码
    :return: tf.data.Dataset 实例以及对应的数据集中元素数量
    """

    dataset_type = dataset_config['type']
    try:
        size = len(dataset_config['src'])
    except:
        size = None

    if dataset_type == 0:
        # 通过 file_path 获取 dataset
        return _get_images_path_dataset(dataset_config), size
    elif dataset_type == 1:
        # 通过分类标签获取 dataset
        return _get_classification_labels_dataset(dataset_config), size
    elif dataset_type == 2:
        # 通过图像分割标签 获取dataset
        return _get_segmentation_labels_dataset(dataset_config), size
    else:
        raise ValueError('unknown dataset type {}'.format(dataset_type))


################################### 根据图片路径获取对应的dataset ###############################################


def get_images_dataset_by_paths_config(file_paths, **kwargs):
    """
    通过file_paths获取图片
    :param file_paths: 图片path
    :param kwargs: 举例如下
    {
        'norm_fn_first': None,
        'norm_fn_end': None,
        'random_flip_horizontal_flag': False,
        'random_flip_vertical_flag': False,
        'random_distort_color_flag': False,
        'distort_color_fast_mode_flag': False,

        # 下面列举切片相关配置

        # 无切片(默认配置)
        'crop_type': CropType.no_crop,
        'image_width': None,
        'image_height': None,

        # 中心切片
        'crop_type': CropType.central_crop,
        'crop_width': 300,
        'crop_height': 200,
        'image_width': 500,
        'image_height': 500,

        # 普通随机切片
        'crop_type': CropType.random_normal,
        'crop_width': None,
        'crop_height': None,
        'image_width': None,
        'image_height': None,

        # vgg随机切片
        'crop_type': CropType.random_vgg,
        'crop_width': None,
        'crop_height': None,
        'vgg_image_size_min': None,
        'vgg_image_size_max': None,

        # inception随机切片
        'crop_type': CropType.random_inception,
        'crop_width': None,
        'crop_height': None,
        'inception_bbox': None,

    }
    :return:
    """
    dataset_config = {
        'type': 0,
        'src': file_paths,
        'norm_fn_first': kwargs.get('norm_fn_first'),
        'norm_fn_end': kwargs.get('norm_fn_end'),
        'random_flip_horizontal_flag': kwargs.get('random_flip_horizontal_flag') or False,
        'random_flip_vertical_flag': kwargs.get('random_flip_vertical_flag') or False,
        'random_distort_color_flag': kwargs.get('random_distort_color_flag') or False,
        'distort_color_fast_mode_flag': kwargs.get('distort_color_fast_mode_flag') or False,

        'crop_type': kwargs.get('crop_type') or CropType.no_crop,
        'image_width': kwargs.get('image_width'),
        'image_height': kwargs.get('image_height'),
        'crop_width': kwargs.get('crop_width'),
        'crop_height': kwargs.get('crop_height'),
        'vgg_image_size_min': kwargs.get('vgg_image_size_min'),
        'vgg_image_size_max': kwargs.get('vgg_image_size_max'),
        'inception_bbox': kwargs.get('inception_bbox'),
    }
    return dataset_config


class CropType(Enum):
    no_crop = 1
    central = 2
    random_normal = 3
    random_vgg = 4
    random_inception = 5


def _get_images_path_dataset(dataset_config):
    """
    通过图片路径以及数据增强参数，获取对应的 dataset
    :param dataset_config: 参考本文件中 get_image_by_path_dataset_config 方法
    :return: tf.data.Dataset 实例
    """
    file_paths = dataset_config.get('src')  # 原始数据，即图片路径
    norm_fn_first = dataset_config.get('norm_fn_first')  # 归一化函数
    norm_fn_end = dataset_config.get('norm_fn_end')  # 归一化函数

    # 切片参数
    crop_type = dataset_config.get('crop_type')  # crop方法
    # 可以选择 无切片、普通随机切片、中心切片、vgg切片、inception切片四种
    crop_width = dataset_config.get('crop_width')  # 切片宽
    crop_height = dataset_config.get('crop_height')  # 切片高
    image_width = dataset_config.get('image_width')  # 用于无切片、普通切片、中心切片
    image_height = dataset_config.get('image_height')  # 用于无切片、普通切片、中心切片
    vgg_image_size_min = dataset_config.get('vgg_image_size_min')  # 用于vgg切片
    vgg_image_size_max = dataset_config.get('vgg_image_size_max')  # 用于vgg切片
    inception_bbox = dataset_config.get('inception_bbox')  # 用于inception切片

    random_flip_horizontal_flag = dataset_config.get('random_flip_horizontal_flag')  # 随机水平镜像
    random_flip_vertical_flag = dataset_config.get('random_flip_vertical_flag')  # 随机垂直镜像
    random_distort_color_flag = dataset_config.get('random_distort_color_flag')  # 随机颜色变换
    distort_color_fast_mode_flag = dataset_config.get('distort_color_fast_mode_flag')  # 颜色变换模式

    def _cur_parse_image_fn(image_path):
        img_file = tf.read_file(image_path)
        # 由于以下issue所描述的问题，不能使用decode_image，而要使用decode_jpeg
        # https://github.com/tensorflow/tensorflow/issues/14226
        cur_image = tf.image.decode_jpeg(img_file, channels=3)

        if norm_fn_first:
            cur_image = norm_fn_first(cur_image)

        # 镜像
        if random_flip_horizontal_flag:
            cur_image = tf.image.random_flip_left_right(cur_image)
        if random_flip_vertical_flag:
            cur_image = tf.image.random_flip_up_down(cur_image)

        # resize与切片
        if crop_type is CropType.no_crop:
            if image_width is not None and image_height is not None:
                cur_image = tf.image.resize_images(cur_image, [image_height, image_width])
        elif crop_type is CropType.central:
            if crop_width is None or crop_height is None:
                raise ValueError('crop_width and crop_height must not be None when using central crop')
            if image_width is not None and image_height is not None:
                cur_image = tf.image.resize_images(cur_image, [image_height, image_width])
            cur_image = central_crop(cur_image, crop_height, crop_width)
        elif crop_type is CropType.random_normal:
            if crop_width is None or crop_height is None:
                raise ValueError('crop_width and crop_height must not be None when using normal random crop')
            if image_width is None or image_height is None:
                raise ValueError('image_width and image_height must not be None when using normal random crop')
            cur_image = tf.image.resize_images(cur_image, [image_height, image_width])
            cur_image = random_crop(cur_image, crop_height, crop_width)
        elif crop_type is CropType.random_vgg:
            if crop_width is None or crop_height is None:
                raise ValueError('crop_width and crop_height must not be None when using vgg random crop')
            if vgg_image_size_max is None or vgg_image_size_min is None:
                raise ValueError('vgg_image_size_min and vgg_image_size_max '
                                 'must not be None when using vgg random crop')
            cur_image = random_crop_vgg(cur_image,
                                        vgg_image_size_min, vgg_image_size_max,
                                        crop_height, crop_width)
        elif crop_type is CropType.random_inception:
            if crop_width is None or crop_height is None:
                raise ValueError('crop_width and crop_height must not be None when using vgg random crop')
            cur_image = random_crop_inception(cur_image, crop_height, crop_width, inception_bbox)
        else:
            raise ValueError('undown crop type {}'.format(crop_type))

        # 色彩变换
        if random_distort_color_flag:
            cur_image = random_distort_color(cur_image, distort_color_fast_mode_flag)

        if norm_fn_end:
            cur_image = norm_fn_end(cur_image)

        return cur_image

    cur_dataset = tf.data.Dataset.from_tensor_slices(file_paths).map(_cur_parse_image_fn)
    return cur_dataset


################################### 根据分类标签获取对应的dataset ###############################################

def get_classification_labels_dataset_config(labels):
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


def _get_classification_labels_dataset(dataset_config):
    """
    {
        'type': 1,
        'src': []
    }
    :param dataset_config:
    :return:
    """
    src = dataset_config['src']
    return tf.data.Dataset.from_tensor_slices(src)


###################################### 获取 图像分割 标签 ##################################################

def get_segmentation_labels_dataset_config(file_paths, color_to_int_list, **kwargs):
    return {
        'type': 2,
        'src': file_paths,
        'color_to_int_list': color_to_int_list,
        'image_height': kwargs.get('image_height'),
        'image_width': kwargs.get('image_width'),
    }


def _get_segmentation_labels_dataset(dataset_config):
    file_paths = dataset_config.get('src')  # 原始数据，即图片路径
    color_to_int_list = dataset_config.get('color_to_int_list')  # 归一化函数
    image_height = dataset_config.get('image_height')
    image_width = dataset_config.get('image_width')

    def _cur_parse_image_fn(image_path):
        img_file = tf.read_file(image_path)
        cur_image = tf.image.decode_jpeg(img_file, channels=3)

        if image_width is not None and image_height is not None:
            cur_image = tf.expand_dims(cur_image, 0)
            cur_image = tf.image.resize_nearest_neighbor(cur_image, [image_height, image_width])
            cur_image = tf.squeeze(cur_image, [0])

        channels = tf.split(cur_image, 3, axis=2)
        cur_image = (256 * channels[0] + channels[1]) * 256 + channels[2]
        return tf.squeeze(tf.cast(tf.gather(color_to_int_list, tf.cast(cur_image, tf.int32)), tf.int32), axis=[-1])

    return tf.data.Dataset.from_tensor_slices(file_paths).map(_cur_parse_image_fn)
