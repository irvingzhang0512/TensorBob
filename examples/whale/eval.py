import tensorbob as bob
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging
from nets import nets_factory

logger = logging.getLogger('tensorflow')
logger.setLevel(logging.DEBUG)

DATA_PATH = '/home/tensorflow05/data/kaggle/humpback_whale_identification'
NUM_CLASSES = 4251


def get_image_file_names_and_paths(data_path):
    sample_submission_file = os.path.join(data_path, 'sample_submission.csv')
    test_dir = os.path.join(data_path, 'test')
    with open(sample_submission_file, 'r') as f:
        raw_lines = f.readlines()
    file_names = [line[:line.find(',')] for line in raw_lines]
    image_paths = [os.path.join(test_dir, file_name) for file_name in file_names]
    return file_names, image_paths


def get_label_dict(data_path):
    train_csv_file = os.path.join(data_path, 'train.csv')
    df = pd.read_csv(train_csv_file)
    labels_str_to_int = {}
    labels_int_to_str = {}
    cur_id = 0
    for c, group in df.groupby("Id"):
        labels_str_to_int[c] = cur_id
        labels_int_to_str[cur_id] = c
        cur_id += 1

    return labels_str_to_int, labels_int_to_str


class WhaleEvaluator(bob.evaluator.Evaluator):
    def __init__(self, data_path, **kwargs):
        self.file_names, self.image_paths = get_image_file_names_and_paths(data_path)
        super().__init__(with_labels=False, **kwargs)

    def _get_test_dataset(self):
        ph_val_image_size = tf.placeholder(tf.int32)
        test_config = {
            'norm_fn_first': bob.data.norm_zero_to_one,
            'norm_fn_end': bob.data.norm_minus_one_to_one,
            'image_width': ph_val_image_size,
            'image_height': ph_val_image_size,
        }
        test_config = bob.data.get_images_dataset_by_paths_config(self.image_paths, **test_config)
        test_dataset = bob.data.BaseDataset([test_config],
                                            batch_size=self._batch_size,
                                            shuffle=False,
                                            repeat=False,
                                            prefetch_buffer_size=1000)
        return test_dataset, ph_val_image_size

    def _get_graph_and_feed_dict(self):
        ph_x = tf.placeholder(tf.float32)
        ph_image_size = tf.placeholder(tf.int32)
        model_fn = nets_factory.get_network_fn('resnet_v2_50', NUM_CLASSES, weight_decay=0.00005, is_training=False)
        _, end_points = model_fn(tf.reshape(ph_x, [-1, ph_image_size, ph_image_size, 3]))
        return ph_x, ph_image_size, end_points['predictions'], None

    def _get_init_fn(self):
        variables_to_restore = bob.variables.get_variables_to_restore(include=['resnet_v2_50'])
        logger.debug('restore %d vars' % len(variables_to_restore))
        return bob.variables.assign_from_checkpoint_fn(self._pre_trained_model_path,
                                                       variables_to_restore,
                                                       ignore_missing_vars=True,
                                                       reshape_variables=True)


if __name__ == '__main__':
    evaluator_config = {
        "batch_size": 32,
        "multi_scale_list": [299, 384],
        "pre_trained_model_path": '/home/tensorflow05/zyy/tensorbob/examples/whale/logs/best_val/model.ckpt-15000',
    }
    e = WhaleEvaluator(data_path=DATA_PATH, **evaluator_config)

    file_names = e.file_names
    labels_str_to_int, labels_int_to_str = get_label_dict(DATA_PATH)
    predictions = e.evaluate()

    assert predictions.shape[0] == len(file_names)

    ids1 = []
    ids2 = []
    indexes1 = np.argsort(-predictions)[, :5]
    indexes2 = np.argsort(-predictions)

    for i in range(len(file_names)):
        indexes1 = np.argsort(-predictions[i])[:5]
        res1 = ''
        for cur_id, cur_index in enumerate(indexes1):
            res1 = res1 + labels_int_to_str[cur_index]
            if cur_id != 4:
                res1 = res1 + ' '
        ids1.append(res1)

        indexes2 = np.argsort(-predictions[i][1:])[:4]
        res2 = 'new_whale'
        for cur_index in indexes2:
            res2 = res2 + ' ' + labels_int_to_str[cur_index]
        ids2.append(res2)

    csv1 = pd.DataFrame({
        'Image': file_names,
        'Id': ids1
    })
    csv2 = pd.DataFrame({
        'Image': file_names,
        'Id': ids2
    })
    csv1.to_csv('./res1.csv', index=False, columns=['Image', 'Id'])
    csv2.to_csv('./res2.csv', index=False, columns=['Image', 'Id'])
