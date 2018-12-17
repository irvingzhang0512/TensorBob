# coding=utf-8
import tensorflow as tf
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pickle
import numpy as np


def generate_k_folds(k_folds, labels):
    placeholder_x = np.zeros(labels.shape[0])
    msss = MultilabelStratifiedKFold(n_splits=k_folds)
    i = 0
    for train_index, val_index in msss.split(placeholder_x, labels):
        with open('%d.pkl' % i, 'wb') as f:
            pickle.dump((train_index, val_index), f)
        i += 1


def get_k_folds(k_folds_index):
    with open('%d.pkl' % k_folds_index, 'rb') as f:
        return pickle.load(f)


def focal_loss(logits, labels, alpha=0.25, gamma=2):
    with tf.variable_scope('focal_loss'):
        sigmoid_p = tf.nn.sigmoid(logits)
        cur_labels = tf.cast(labels, tf.float32)

        pt = sigmoid_p * cur_labels + (1 - sigmoid_p) * (1 - cur_labels)
        w = alpha * cur_labels + (1 - alpha) * (1 - cur_labels)
        w = w * tf.pow((1 - pt), gamma)

        return tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits, weights=w,
                                               scope='focal_loss_partial')


def focal_loss_v2(logits, labels, alpha=0.25, gamma=2):
    y_true = tf.cast(labels, tf.float32)
    sigmoid_p = tf.nn.sigmoid(logits)
    y_pred = tf.cast(tf.greater(tf.cast(sigmoid_p, tf.float32), thresholds), tf.float32)

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    y_pred = tf.log(y_pred / (1 - y_pred))

    input = tf.cast(y_pred, tf.float32)

    max_val = tf.clip_by_value(-input, 0, 1)
    loss = input - input * y_true + max_val + tf.log(tf.exp(-max_val) + tf.exp(-input - max_val))
    invprobs = tf.log_sigmoid(-input * (y_true * 2.0 - 1.0))
    loss = tf.exp(invprobs * gamma) * loss

    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))


def f1_loss(y_true, y_pred, threshold):
    with tf.variable_scope('f1_loss'):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.greater(tf.cast(y_pred, tf.float32), threshold), tf.float32)
        tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32), axis=0)
        tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.float32), axis=0)
        fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32), axis=0)
        fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32), axis=0)

        p = tp / (tp + fp + 1e-7)
        r = tp / (tp + fn + 1e-7)

        f1 = 2 * p * r / (p + r + 1e-7)
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        cur_loss = 1 - tf.reduce_mean(f1)
        tf.losses.add_loss(cur_loss)
        return cur_loss


def fbeta_score_macro(y_true, y_pred, beta=1, threshold=None):
    # https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.greater(tf.cast(y_pred, tf.float32), threshold), tf.float32)

    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)

    f1 = (1 + beta ** 2) * p * r / ((beta ** 2) * p + r + 1e-7)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return tf.reduce_mean(f1)
