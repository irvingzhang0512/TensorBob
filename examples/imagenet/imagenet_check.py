import os
import cv2

# DATA_PATH = "H:\\ILSVRC2012"
DATA_PATH = "/home/tensorflow05/data/ILSVRC2012"
TRAIN_PATH = os.path.join(DATA_PATH, 'ILSVRC2012_img_train')
VAL_PATH = os.path.join(DATA_PATH, 'ILSVRC2012_img_val')


def check_train():
    f = open('broken_images_train.txt', 'w')
    for wnid in os.listdir(TRAIN_PATH):
        cur_dir = os.path.join(TRAIN_PATH, wnid)
        for pic in os.listdir(cur_dir):
            cur_pic = os.path.join(cur_dir, pic)
            try:
                img = cv2.imread(cur_pic)
                if img.shape[2] != 3:
                    f.write(pic + '\n')
            except:
                f.write(pic + '\n')
    f.close()


def check_val():
    f = open('broken_images_val.txt', 'w')
    for img in os.listdir(VAL_PATH):
        cur_pic = os.path.join(VAL_PATH, img)
        try:
            img = cv2.imread(cur_pic)
            if img.shape[2] != 3:
                f.write(img + '\n')
        except:
            f.write(img + '\n')
    f.close()


if __name__ == '__main__':
    check_train()
    check_val()
