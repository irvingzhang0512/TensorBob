import tensorflow as tf
import tensorbob as bob
import logging
import numpy as np
from nets import nets_factory

logger = logging.getLogger('tensorflow')
logger.setLevel(logging.DEBUG)


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
                                                                    data_path='/home/tensorflow05/data/ILSVRC2012',
                                                                    crop_type=bob.data.CropType.no_crop,
                                                                    image_width=ph_val_image_size,
                                                                    image_height=ph_val_image_size,
                                                                    # # vgg_16
                                                                    # norm_fn_first=bob.data.norm_imagenet,
                                                                    # labels_offset=0,

                                                                    # inception_v3
                                                                    norm_fn_first=bob.data.norm_zero_to_one,
                                                                    norm_fn_end=bob.data.norm_minus_one_to_one,
                                                                    labels_offset=1,
                                                                    )
        return test_dataset, ph_val_image_size

    def _get_graph_and_feed_dict(self):
        ph_x = tf.placeholder(tf.float32)
        ph_image_size = tf.placeholder(tf.int32)
        # model_fn = nets_factory.get_network_fn('vgg_16', 1000)
        model_fn = nets_factory.get_network_fn('inception_v3', 1001)
        _, end_points = model_fn(tf.reshape(ph_x, [-1, ph_image_size, ph_image_size, 3]), global_pool=True)
        return ph_x, ph_image_size, end_points['Predictions'], None

    def _get_init_fn(self):
        # variables_to_restore = get_variables_to_restore(include=['vgg_16'])
        variables_to_restore = bob.variables.get_variables_to_restore(include=['InceptionV3'])
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
e = bob.evaluator.ImageNetEvaluator(batch_size=32,
                                    # multi_scale_list=[512, 384, 299],
                                    evaluate_image_size=299,
                                    pre_trained_model_path='/home/tensorflow05/data/pre-trained/slim/inception_v3.ckpt'
                                    )

predictions, labels = e.evaluate()
accuracy = np.mean(np.equal(np.argmax(predictions, axis=1), labels).astype(np.float32))

logger.info(str(accuracy))
