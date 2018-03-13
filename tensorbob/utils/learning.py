from tensorflow.contrib.slim.python.slim.learning import *
import time


def get_train_step_kwargs(dataset, placeholders, should_log, number_of_steps):
    return {'dataset': dataset,
            'placeholders': placeholders,
            'should_log': should_log,
            'number_of_steps': number_of_steps}


def train_step_bob(sess, train_op, global_step, train_step_kwargs):
    if 'dataset' not in train_step_kwargs:
        raise ValueError('must have dataset in train_step_kwargs!')
    dataset = train_step_kwargs['dataset']
    current_batch = dataset.get_next_batch(sess)

    if 'placeholders' not in train_step_kwargs:
        raise ValueError('must have placeholders in train_step_kwargs!')
    placeholders = train_step_kwargs['placeholders']
    assert len(placeholders) == len(current_batch)
    feed_dict = {key: value for key, value in zip(placeholders, current_batch)}

    start_time = time.time()
    total_loss, np_global_step = sess.run([train_op, global_step], feed_dict=feed_dict)
    time_elapsed = time.time() - start_time

    print('global step %d: loss = %.4f (%.3f sec/step)',
          np_global_step, total_loss, time_elapsed)

    if np_global_step >= train_step_kwargs['number_of_steps']:
        should_stop = True
    else:
        should_stop = False

    return total_loss, should_stop
