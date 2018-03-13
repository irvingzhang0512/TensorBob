from .voc2012 import get_voc_classification_dataset
import tensorflow as tf

if __name__ == '__main__':
    train_dataset = get_voc_classification_dataset('train')
    val_dataset = get_voc_classification_dataset('val')

    with tf.Session() as sess:
        i = 0
        while True:
            images, labels = train_dataset.get_next_batch(sess)
            print(images.shape, labels, i)
            i += 32
