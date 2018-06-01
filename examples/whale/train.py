import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorbob as bob
from nets import nets_factory
import logging
logger = logging.getLogger('tensorflow')
logger.setLevel(logging.DEBUG)

NUM_CLASSES = 4251


def get_train_val_paths_and_labels(data_root):
    train_csv_file = os.path.join(data_root, 'train.csv')
    train_dir = os.path.join(data_root, 'train')

    df = pd.read_csv(train_csv_file)
    labels_str_to_int = {}
    train_paths = np.array([], dtype=np.string_)
    train_labels = np.array([], dtype=np.int32)
    val_paths = np.array([], dtype=np.string_)
    val_labels = np.array([], dtype=np.int32)

    cur_id = 0
    for c, group in df.groupby("Id"):
        labels_str_to_int[c] = cur_id
        images = group['Image'].values
        images = np.array([os.path.join(train_dir, image) for image in images])
        val_labels = np.append(val_labels, cur_id)
        val_paths = np.append(val_paths, images[0])

        if len(images) != 1:
            images = images[1:]
        while len(images) < 8:
            images = np.append(images, images)

        train_paths = np.append(train_paths, images)
        train_labels = np.append(train_labels, [cur_id for _ in range(len(images))])

        cur_id += 1
    assert len(val_paths) == len(val_labels)
    assert len(train_paths) == len(train_labels)

    return train_paths, train_labels, val_paths, val_labels, labels_str_to_int


class WhaleTrainer(bob.basic_trainers.BaseClassificationTrainer):
    def __init__(self, data_root='D:\\PycharmProjects\\data\\kaggle\\humpback_whale_identification',
                 pre_trained_model_path=None,
                 **kwargs):
        # {
        #     'training_crop_size': 224,
        #     'val_crop_size': 384,
        #     'fine_tune_steps': None,
        #     'fine_tune_var_list': None,
        #
        #     'batch_size': 32,
        #     'weight_decay': 0.00005,
        #     'keep_prob': 0.5,
        #     'learning_rate_start': 0.001,
        #     'lr_decay_rate': 0.5,
        #     'lr_decay_steps': 40000 * 10,
        #     'lr_staircase': False,
        #     'step_ckpt_dir': './logs/ckpt/',
        #     'train_logs_dir': './logs/train/',
        #     'val_logs_dir': './logs/val/',
        #     'best_val_ckpt_dir': './logs/best_val/',
        #     'metrics_collection': 'val_metrics',
        #     'metrics_update_ops_collection': 'update_ops',
        #     'metrics_reset_ops_collection': 'reset_ops',
        #     'use_mean_metrics': False,
        #     'logging_every_n_steps': 1000,
        #     'summary_every_n_steps': 1000,
        #     'save_every_n_steps': 1000,
        #     'evaluate_every_n_steps': 10000,
        #     'max_steps': None
        # }
        logger.debug('whale kwargs is {}'.format(kwargs))
        super().__init__(num_classes=4251, **kwargs)
        self._pre_trained_model_path = pre_trained_model_path
        self._train_paths, self._train_labels, self._val_paths, self._val_labels, self._labels_str_to_int = get_train_val_paths_and_labels(
            data_root)

    def _get_scaffold(self):
        if self._pre_trained_model_path is not None:
            variables_to_restore = bob.variables.get_variables_to_restore(include=['InceptionV3'],
                                                                          exclude=['InceptionV3/Logits'])
            init_fn = bob.variables.assign_from_checkpoint_fn(self._pre_trained_model_path,
                                                              variables_to_restore,
                                                              ignore_missing_vars=True,
                                                              reshape_variables=True)

            def new_init_fn(scaffold, session):
                init_fn(session)

            return tf.train.Scaffold(init_fn=new_init_fn)

    def _get_training_dataset(self):
        train_configs = {
            'norm_fn_first': bob.data.norm_zero_to_one,
            'norm_fn_end': bob.data.norm_minus_one_to_one,
            'crop_type': bob.data.CropType.random_inception,
            'crop_width': self._training_crop_size,
            'crop_height': self._training_crop_size,
            'inception_bbox': None,

            'random_flip_horizontal_flag': True,
            'random_distort_color_flag': True,
        }
        images_config = bob.data.get_images_dataset_by_paths_config(self._train_paths, **train_configs)
        labels_config = bob.data.get_classification_labels_dataset_config(self._train_labels)
        dataset_config = [images_config, labels_config]
        return bob.data.BaseDataset(dataset_config,
                                    batch_size=self._batch_size,
                                    shuffle=False,
                                    repeat=True,
                                    prefetch_buffer_size=10000)

    def _get_val_dataset(self):
        val_configs = {
            'norm_fn_first': bob.data.norm_zero_to_one,
            'norm_fn_end': bob.data.norm_minus_one_to_one,
            'image_width': self._val_crop_size,
            'image_height': self._val_crop_size,
        }
        images_config = bob.data.get_images_dataset_by_paths_config(self._val_paths, **val_configs)
        labels_config = bob.data.get_classification_labels_dataset_config(self._val_labels)
        dataset_configs = [images_config, labels_config]
        return bob.data.BaseDataset(dataset_configs,
                                    batch_size=self._batch_size,
                                    shuffle=False,
                                    repeat=False)

    def _get_model(self):
        network_fn = nets_factory.get_network_fn('inception_v3',
                                                 num_classes=self._num_classes,
                                                 weight_decay=self._weight_decay,
                                                 is_training=self._ph_is_training,
                                                 )
        return network_fn(images=tf.reshape(self._ph_x, [-1, self._ph_image_size, self._ph_image_size, 3]),
                          dropout_keep_prob=self._keep_prob,
                          global_pool=True,
                          create_aux_logits=False
                          )

    def _get_optimizer(self):
        return tf.train.MomentumOptimizer(learning_rate=self._get_learning_rate(),
                                          momentum=0.9)


if __name__ == '__main__':
    whale = WhaleTrainer(
        data_root='/home/ubuntu/data/kaggle/humpback',
        pre_trained_model_path='/home/ubuntu/data/slim/inception_v3.ckpt',
        training_crop_size=299,
        best_val_ckpt_dir='./logs/best_val/model.ckpt',
        logging_every_n_steps=100,
        summary_every_n_steps=100,
        save_every_n_steps=4000,
        evaluate_every_n_steps=1000,
        lr_decay_steps=5000,
        lr_staircase=True,
        learning_rate_start=0.0005
    )
    whale.train()

