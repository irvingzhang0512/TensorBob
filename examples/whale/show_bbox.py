import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# 之前代码没注意，顺序搞错了
with open('./locations_train.pickle', 'rb') as infile:
    bboxes_old = pickle.load(infile)
bboxes = np.zeros(bboxes_old.shape)
bboxes[:, 0] = bboxes_old[:, 1]
bboxes[:, 1] = bboxes_old[:, 0]
bboxes[:, 2] = bboxes_old[:, 3]
bboxes[:, 3] = bboxes_old[:, 2]

# 读取文件路径
df = pd.read_csv('/home/tensorflow05/data/kaggle/humpback_whale_identification/train.csv')
cur_id = -1
paths = []
for c, group in df.groupby("Id"):
    if cur_id == -1:
        cur_id += 1
        continue
    images = group['Image'].values
    images = [os.path.join("/home/tensorflow05/data/kaggle/humpback_whale_identification/train", image)
              for image in images]
    paths += images
len(paths)

# 直接用 tf.image.draw_bounding_boxes 边框好像有问题
# 所以就直接获取图片，并显示
ph_file_name = tf.placeholder(tf.string)
ph_bbox = tf.placeholder(tf.float32, [1, 1, 4])
img = tf.read_file(ph_file_name)
img = tf.image.decode_jpeg(img, 3)
batched = tf.expand_dims(tf.image.convert_image_dtype(img, tf.float32), 0)
bbox = tf.cast(ph_bbox, tf.float32)
final_image = tf.image.draw_bounding_boxes(batched, bbox)
with tf.Session() as sess:
    for i in range(50, 80):
        feed_dict = {ph_file_name: paths[i],
                     ph_bbox: bboxes[i].reshape(1, 1, 4)}
        cur_img, cur_final_img = sess.run([batched, final_image], feed_dict=feed_dict)
        cur_img = np.squeeze(cur_img, 0)
        cur_final_img = np.squeeze(cur_final_img, 0)
        plt.imshow(cur_final_img)
        plt.show()
        image_height, image_width, channels = cur_img.shape
        ymin, xmin, ymax, xmax = bboxes[i]
        sliced_img = cur_img[
                     int(image_height * ymin): int(image_height * ymax),
                     int(image_width * xmin): int(image_width * xmax),
                     :]
        plt.imshow(sliced_img)
        plt.show()
