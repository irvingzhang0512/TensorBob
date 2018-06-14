import tensorflow as tf
import tensorbob as bob
import logging
import numpy as np
from nets import nets_factory

logger = logging.getLogger('tensorflow')
logger.setLevel(logging.DEBUG)

DATA_PATH = '/home/tensorflow05/data/ILSVRC2012'

PRE_TRAINED_MODEL_PATH = '/home/tensorflow05/data/pre-trained/slim/inception_v3.ckpt'
NUM_CLASSES = 1001
LABELS_OFFSET = NUM_CLASSES - 1000
NORM_FN_FIRST = bob.data.norm_zero_to_one
NORM_FN_END = bob.data.norm_minus_one_to_one
MODEL_NAME = 'inception_v3'
PREDICTION_END_POINT_KEY = 'Predictions'
VAR_INCLUDE = ['InceptionV3']


class ImageNetEvaluator(bob.evaluator.Evaluator):
    def __init__(self, **kwargs):
        """
        batch_size=32,
        multi_crop_number=None,
        multi_scale_list=None,
        evaluate_image_size=None

        :param kwargs:  参数如上
        """
        super().__init__(**kwargs)

    def _get_test_dataset(self):
        ph_val_image_size = tf.placeholder(tf.int32)
        test_dataset = bob.data.get_imagenet_classification_dataset(mode='val',
                                                                    batch_size=self._batch_size,
                                                                    data_path=DATA_PATH,
                                                                    crop_type=bob.data.CropType.no_crop,
                                                                    image_width=ph_val_image_size,
                                                                    image_height=ph_val_image_size,
                                                                    norm_fn_first=NORM_FN_FIRST,
                                                                    norm_fn_end=NORM_FN_END,
                                                                    labels_offset=LABELS_OFFSET,
                                                                    )
        return test_dataset, ph_val_image_size

    def _get_graph_and_feed_dict(self):
        ph_x = tf.placeholder(tf.float32)
        ph_image_size = tf.placeholder(tf.int32)
        model_fn = nets_factory.get_network_fn(MODEL_NAME, NUM_CLASSES)
        _, end_points = model_fn(tf.reshape(ph_x, [-1, ph_image_size, ph_image_size, 3]))
        return ph_x, ph_image_size, end_points[PREDICTION_END_POINT_KEY], None

    def _get_init_fn(self):
        variables_to_restore = bob.variables.get_variables_to_restore(include=VAR_INCLUDE)
        logging.debug('restore %d vars' % len(variables_to_restore))
        return bob.variables.assign_from_checkpoint_fn(self._pre_trained_model_path,
                                                       variables_to_restore,
                                                       ignore_missing_vars=True,
                                                       reshape_variables=True)


# vgg 16
# 384 0.71116
# [224, 256, 384] 0.7026

# inception v3
# 299 0.7694
# [299, 384, 512] 0.7827
e = ImageNetEvaluator(batch_size=32,
                                    # multi_scale_list=[512, 384, 299],
                                    evaluate_image_size=331,
                                    pre_trained_model_path=PRE_TRAINED_MODEL_PATH
                                    )

predictions, labels = e.evaluate()
accuracy = np.mean(np.equal(np.argmax(predictions, axis=1), labels).astype(np.float32))

logger.info(str(accuracy))
