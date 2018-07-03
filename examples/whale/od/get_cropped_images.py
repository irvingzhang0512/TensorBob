import numpy as np
import os
import tensorflow as tf
from PIL import Image
from object_detection.utils import label_map_util

PATH_TO_CKPT = '/home/ubuntu/bob/models/research/object_detection/logs_whale_resnet101_export/frozen_inference_graph.pb'
PATH_TO_LABELS = "/home/ubuntu/bob/TensorBob/examples/whale/od/label_map.pbtxt"
FROM_DIR = '/home/ubuntu/data/kaggle/humpback/train'
TO_DIR = '/home/ubuntu/data/kaggle/humpback/train_crop'
NUM_CLASSES = 1
IMAGE_NAMES = os.listdir(FROM_DIR)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def crop_images(session):
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    for idx, image_name in enumerate(IMAGE_NAMES):
        if idx % 100 == 0:
            print('cropped %d/%d images' % (idx, len(IMAGE_NAMES)))
        full_image_path = os.path.join(FROM_DIR, image_name)
        raw_image = Image.open(full_image_path).convert('RGB')
        image_np = np.array(raw_image)
        image = np.expand_dims(image_np, axis=0)

        output_dict = session.run(tensor_dict, feed_dict={image_tensor: image})
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]

        if output_dict['num_detections'] != 1:
            raw_image.save(os.path.join(TO_DIR, image_name))
            print('num_detections of picture {} is not 1!'.format(image_name))
            continue

        ymin, xmin, ymax, xmax = output_dict['detection_boxes'][0]
        image_height, image_width, num_channels = image_np.shape
        ymin = int(image_height * ymin)
        xmin = int(image_width * xmin)
        ymax = int(image_height * ymax)
        xmax = int(image_width * xmax)
        raw_image.crop((xmin, ymin, xmax, ymax)).save(os.path.join(TO_DIR, image_name))


with detection_graph.as_default():
    with tf.Session() as sess:
        crop_images(sess)
