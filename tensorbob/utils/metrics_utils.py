import tensorflow as tf


def compute_mean_iou_by_confusion_matrix(name, total_cm):
    with tf.variable_scope('compute_mean_iou_by_confusion_matrix'):
        sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
        sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
        cm_diag = tf.to_float(tf.diag_part(total_cm))
        denominator = sum_over_row + sum_over_col - cm_diag

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(
                tf.not_equal(denominator, 0), dtype=tf.float32))

        # If the value of the denominator is 0, set it to 1 to avoid
        # zero division.
        denominator = tf.where(
            tf.greater(denominator, 0), denominator,
            tf.ones_like(denominator))
        iou = tf.div(cm_diag, denominator)

        # If the number of valid entries is 0 (no classes) we return 0.
        result = tf.where(
            tf.greater(num_valid_entries, 0),
            tf.reduce_sum(iou, name=name) / num_valid_entries, 0)
        return result


def compute_mean_iou(name, predictions, labels):
    with tf.variable_scope('compute_mean_iou'):
        total_cm = tf.confusion_matrix(predictions=predictions, labels=labels)
        return compute_mean_iou_by_confusion_matrix(name, total_cm)
