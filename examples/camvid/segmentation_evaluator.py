import tensorbob as bob
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
logging.set_verbosity(logging.DEBUG)


class CamVidEvaluator(bob.evaluating.Evaluator):
    def __init__(self, dataset_configs):
        super().__init__(**dataset_configs)

    def _get_test_dataset(self):
        return bob.data.get_camvid_segmentation_dataset('test',
                                                        batch_size=self._batch_size,
                                                        repeat=1,
                                                        norm_fn_first=bob.preprocessing.norm_zero_to_one,
                                                        norm_fn_end=bob.preprocessing.norm_minus_one_to_one,
                                                        image_width=self._evaluate_image_width,
                                                        image_height=self._evaluate_image_height,
                                                        )

    def _get_metrics_and_feed_dict(self):
        x, y = self._next_batch
        logits, _ = bob.segmentation.fc_densenet(x,
                                                 num_classes=32,
                                                 is_training=False,
                                                 keep_prob=1.0,
                                                 weight_decay=0.0001)
        predictions = tf.argmax(logits, axis=-1)
        _, accuracy = tf.metrics.accuracy(y, predictions,
                                          name='accuracy')
        _, confused_matrix = tf.metrics.mean_iou(tf.reshape(y, [-1]),
                                                 tf.reshape(predictions, [-1]),
                                                 32,
                                                 name='confused_matrix')
        mean_iou = bob.metrics_utils.compute_mean_iou_by_confusion_matrix('mean_iou', confused_matrix)
        _, accuarcy_per_class = tf.metrics.mean_per_class_accuracy(y, predictions, 32)

        for metric in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES):
            tf.add_to_collection(self._metrics_reset_ops_collection,
                                 tf.assign(metric, tf.zeros(metric.get_shape(), metric.dtype)))

        return [mean_iou, accuarcy_per_class, accuracy], None

    def _get_init_fn(self):
        def init_fn(sess):
            sess.run(tf.local_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, self._pre_trained_model_path)
            logging.debug('init_fn successfully processed...')

        return init_fn


def main(args):
    evaluator_configs = {
        'batch_size': 1,
        'multi_crop_number': None,
        'multi_scale_list': None,
        'evaluate_image_width': 256,
        'evaluate_image_height': 256,
        'pre_trained_model_path': "E:\\PycharmProjects\\TensorBob\\examples\\camvid\\logs\\val\\model.ckpt-4200",
    }
    evaluator = CamVidEvaluator(evaluator_configs)
    evaluator.evaluate()


main(None)
