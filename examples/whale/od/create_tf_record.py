import os
from lxml import etree
import PIL.Image
import tensorflow as tf
import hashlib
import io
from tensorflow.python.platform import tf_logging as logging
from object_detection.utils import dataset_util

logging.set_verbosity(logging.DEBUG)

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/ubuntu/data/kaggle/humpback', '')
flags.DEFINE_string('output_path', '/home/ubuntu/data/kaggle/humpback/whale_train.record', '')
flags.DEFINE_string('set', 'train', '')
flags.DEFINE_string('images_dir', 'train', '')
flags.DEFINE_string('annotations_dir', 'Annotations', '')
FLAGS = flags.FLAGS

SETS = ['train', 'test']


def dict_to_tf_example(data,
                       dataset_directory):
    full_path = os.path.join(dataset_directory, FLAGS.images_dir, data['filename'])
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    difficult_obj = []
    if 'object' in data:
        for obj in data['object']:
            difficult = False

            difficult_obj.append(int(difficult))
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append('whale'.encode('utf8'))
            classes.append(1)
            truncated.append(0)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
    }))
    return example


def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))
    data_dir = FLAGS.data_dir
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    annotations_dir = os.path.join(data_dir, FLAGS.annotations_dir)

    examples_list = os.listdir(annotations_dir)

    for idx, example in enumerate(examples_list):
        if idx % 50 == 0:
            logging.info('On image %d of %d', idx, len(examples_list))
        path = os.path.join(annotations_dir, example)
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        tf_example = dict_to_tf_example(data, FLAGS.data_dir)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
