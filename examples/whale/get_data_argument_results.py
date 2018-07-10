import pandas as pd
import numpy as np
import tensorflow as tf
import tensorbob as bob
import shutil
import os
from PIL import Image

# csf_file_path = "/home/tensorflow05/data/kaggle/humpback_whale_identification/train.csv"
# csf_file_path_2 = "/home/tensorflow05/data/kaggle/humpback_whale_identification/train_2.csv"
# from_path = "/home/tensorflow05/data/kaggle/humpback_whale_identification/train_crop"
# to_path = "/home/tensorflow05/data/kaggle/humpback_whale_identification/train_crop_2"

csf_file_path = "/home/ubuntu/data/kaggle/humpback/train.csv"
csf_file_path_2 = "/home/ubuntu/data/kaggle/humpback/train_2.csv"
from_path = "/home/ubuntu/data/kaggle/humpback/train_crop"
to_path = "/home/ubuntu/data/kaggle/humpback/train_crop_2"

ph_image_path = tf.placeholder(tf.string)
cur_image = tf.read_file(ph_image_path)
cur_image = tf.image.decode_jpeg(cur_image, channels=3)
raw_image = tf.image.convert_image_dtype(cur_image, tf.float32)
modified_image = bob.data.random_distort_color(raw_image, False)

df = pd.read_csv(csf_file_path)

with tf.Session() as sess:
    for c, group in df.groupby("Id"):
        for image_name in group.Image:
            shutil.copy(os.path.join(from_path, image_name), os.path.join(to_path, image_name))

        original_number = group['Image'].shape[0]
        if original_number < 4:
            for i in range(4 - original_number):
                cur_image_name = group['Image'].values[np.random.randint(original_number)]
                cur_image_path = os.path.join(from_path, cur_image_name)
                cur_image_array = sess.run(modified_image, feed_dict={ph_image_path: cur_image_path})
                dot_idx = cur_image_name.find('.')
                cur_image_name = cur_image_name[:dot_idx] + '_' + str(i) + cur_image_name[dot_idx:]
                df.loc[df.shape[0]] = [cur_image_name, c]
                Image.fromarray((cur_image_array * 255).astype(np.uint8)).save(os.path.join(to_path, cur_image_name))


df.to_csv(csf_file_path_2, index=False, columns=['Image', 'Id'])
print(df.shape)
