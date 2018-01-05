"""
Preprocess image data for Dogs vs. Cats competition
https://www.kaggle.com/gauss256

DESCRIPTION

Most solutions for this competition will need to deal with the fact that the
training and test images are of varying sizes and aspect ratios, and were taken
under a variety of lighting conditions. It is typical in cases like this to
preprocess the images to make them more uniform.

By default this script will normalize the image luminance and resize to a
square image of side length 224. The resizing preserves the aspect ratio and
adds gray bars as necessary to make them square. The resulting images are
stored in a folder named data224.

The location of the input and output files, and the size of the images, can be
controlled through the processing parameters below.

INSTALLATION

This script has been tested on Ubuntu 14.04 using the Anaconda distribution for
Python 3.5. The only additional requirement is for the pillow library for image
manipulation which can be installed via:
    conda install pillow

Source: https://www.kaggle.com/gauss256/preprocess-images
Modified by: Tran Ngoc Minh (M.N.Tran@ibm.com)
"""
import glob
from multiprocessing import Process
import os
import re

import numpy as np
import PIL
from PIL import Image
import pickle
import bz2
import random
import tensorflow as tf
import cv2 as cv

from autodp.utils.tf_utils import bytes_feature
from autodp.utils.tf_utils import int64_feature


# Processing parameters
SIZE = 200    # for ImageNet models compatibility
TEST_DIR = "../storage/inputs/dogcat/origin/test/"
TRAIN_DIR = "../storage/inputs/dogcat/origin/train/"
BASE_DIR = "../storage/inputs/dogcat/"
NUM_CHANNELS = 3


def natural_key(string_):
    """
    Define sort key that is integer-aware
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def norm_image(img):
    """
    Normalize PIL image. Normalizes luminance to (mean,std)=(0,1), and applies
    a [1%, 99%] contrast stretch.
    """
    img_y, img_b, img_r = img.convert('YCbCr').split()

    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0

    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

    img_nrm = img_ybr.convert('RGB')

    return img_nrm


def resize_image(img, size):
    """
    Resize PIL image. Resizes image to be square with sidelength size. Pads
    with black if needed.
    """
    # Resize
    n_x, n_y = img.size
    if n_y > n_x:
        n_y_new = size
        n_x_new = int(size * n_x / n_y + 0.5)
    else:
        n_x_new = size
        n_y_new = int(size * n_y / n_x + 0.5)

    img_res = img.resize((n_x_new, n_y_new), resample=PIL.Image.BICUBIC)

    # Pad the borders to create a square image
    img_pad = Image.new('RGB', (size, size), (128, 128, 128))
    ulc = ((size - n_x_new) // 2, (size - n_y_new) // 2)
    img_pad.paste(img_res, ulc)

    return img_pad


def prep_images(paths, out_file, bool_train):
    """
    Preprocess images. Reads images in paths, and writes to out_dir.
    """
    writer = bz2.BZ2File(out_file, "wb")
    for count, path in enumerate(paths):
        if path.split(sep="/")[-1].split(sep=".")[0] == "cat":
            label = 0
        elif path.split(sep="/")[-1].split(sep=".")[0] == "dog":
            label = 1
        else:
            label = path.split(sep="/")[-1].split(sep=".")[0]
            if bool_train: raise ValueError("Wrong files !!!")

        img = Image.open(path)
        img_nrm = norm_image(img)
        img_res = resize_image(img_nrm, SIZE) #.convert('L')

        line = {"i": np.reshape(img_res, [SIZE, SIZE, NUM_CHANNELS]),
                "l": np.int32(label)}
        pickle.dump(line, writer, pickle.HIGHEST_PROTOCOL)

    writer.close()


def dist_images(paths, out_file, bool_train):
    """
    Preprocess images. Reads images in paths, and writes to out_dir.
    """
    writer = bz2.BZ2File(out_file, "wb")
    for count, path in enumerate(paths):
        if path.split(sep="/")[-1].split(sep=".")[0] == "cat":
            label = 0
        elif path.split(sep="/")[-1].split(sep=".")[0] == "dog":
            label = 1
        else:
            label = path.split(sep="/")[-1].split(sep=".")[0]
            if bool_train: raise ValueError("Wrong files !!!")

        img = process_img(np.array(Image.open(path)))
        img_nrm = norm_image(Image.fromarray(img))
        img_res = resize_image(img_nrm, SIZE) #.convert('L')

        line = {"i": np.reshape(img_res, [SIZE, SIZE, NUM_CHANNELS]),
                "l": np.int32(label)}
        pickle.dump(line, writer, pickle.HIGHEST_PROTOCOL)

    writer.close()


def process_img(img):
    # Process each image
    time = 0
    while (True):
        time += 1
        action = np.random.randint(0, 11)

        if action == 0:
            img = flip(img, -1)

        elif action == 1:
            img = flip(img, 0)

        elif action == 2:
            img = flip(img, 1)

        elif action == 3:
            img = rotate(img, 1)

        elif action == 4:
            img = rotate(img, 2)

        elif action == 5:
            img = rotate(img, 4)

        elif action == 6:
            img = rotate(img, 8)

        elif action == 7:
            img = rotate(img, -1)

        elif action == 8:
            img = rotate(img, -2)

        elif action == 9:
            img = rotate(img, -4)

        elif action == 10:
            img = rotate(img, -8)

        if np.random.rand() <= 0.5 or time == 5: break

    return img


def flip(img, flip_code):
    """
    Flip action.
    :param flip_code:
    :return:
    """
    state = cv.flip(img, flip_code)

    return state


def crop(img, ratio):
    """
    Crop action.
    :param ratio:
    :return:
    """
    crop_size0 = int(img.shape[0] * ratio)
    crop_size1 = int(img.shape[1] * ratio)
    d0 = int((img.shape[0] - crop_size0) / 2)
    d1 = int((img.shape[1] - crop_size1) / 2)

    crop_im = img[d0: d0 + crop_size0, d1: d1 + crop_size1, :]
    state = np.zeros(img.shape)
    state[d0: d0 + crop_size0, d1: d1 + crop_size1, :] = crop_im

    return state


def scale(img, ratio):
    """
    Scale action.
    :param ratio:
    :return:
    """
    height, width = img.shape[:2]
    res_im = cv.resize(img, (int(width * ratio), int(height * ratio)),
                       interpolation=cv.INTER_CUBIC)

    if ratio > 1:
        d0 = int((res_im.shape[0] - height) / 2)
        d1 = int((res_im.shape[1] - width) / 2)
        state = res_im[d0: d0 + height, d1: d1 + width, :]

    elif ratio < 1:
        d0 = int((height - res_im.shape[0]) / 2)
        d1 = int((width - res_im.shape[1]) / 2)
        state = np.zeros(img.shape)
        state[d0:d0 + res_im.shape[0], d1:d1 + res_im.shape[1], :] = res_im

    return state


def rotate(img, degree):
    """
    Rotate action.
    :param degree:
    :return:
    """
    rows, cols = img.shape[:2]
    matrix = cv.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
    state = cv.warpAffine(img, matrix, (cols, rows))

    return state


def main():
    """
    Main program for running from command line.
    """
    # Get the paths to all the image files
    train_cats = sorted(glob.glob(os.path.join(TRAIN_DIR, 'cat*.jpg')),
                        key=natural_key)
    train_dogs = sorted(glob.glob(os.path.join(TRAIN_DIR, 'dog*.jpg')),
                        key=natural_key)
    random.shuffle(train_cats)
    random.shuffle(train_dogs)

    train_all = train_cats[:12000] + train_dogs[:12000]
    valid_all = train_cats[12000:] + train_dogs[12000:]
    random.shuffle(train_all)
    random.shuffle(valid_all)
    random.shuffle(train_all)
    random.shuffle(valid_all)
    random.shuffle(train_all)
    random.shuffle(valid_all)

    test_all = sorted(glob.glob(os.path.join(TEST_DIR, '*.jpg')),
                      key=natural_key)

    # Make the output directories
    base_out = os.path.join(BASE_DIR, "data{}".format(SIZE))
    #train_dir_out = os.path.join(base_out, 'train')
    #test_dir_out = os.path.join(base_out, 'test')
    #os.makedirs(train_dir_out, exist_ok=True)
    #os.makedirs(test_dir_out, exist_ok=True)
    os.makedirs(base_out + "/train", exist_ok=True)
    os.makedirs(base_out + "/valid", exist_ok=True)
    os.makedirs(base_out + "/test", exist_ok=True)
    os.makedirs(base_out + "/dist_test", exist_ok=True)

    # Preprocess the training files
    procs = dict()
    procs[1] = Process(target=prep_images, args=(train_all, os.path.join(
        base_out, "train", "train.bz2"), True, ))
    procs[1].start()
    procs[2] = Process(target=prep_images, args=(valid_all, os.path.join(
        base_out, "valid", "valid.bz2"), True, ))
    procs[2].start()
    procs[3] = Process(target=prep_images, args=(test_all, os.path.join(
        base_out, "test", "test.bz2"), False, ))
    procs[3].start()
    procs[4] = Process(target=dist_images, args=(test_all, os.path.join(
        base_out, "dist_test", "dist_test.bz2"), False, ))
    procs[4].start()

    procs[1].join()
    procs[2].join()
    procs[3].join()
    procs[4].join()


def convert_2tf():
    """
    This function helps convert to tfrecords.
    :return:
    """
    set = ["train", "valid", "test"]

    for s in set:
        ins = os.path.join(BASE_DIR, "data50", s, s + ".bz2")
        outs = os.path.join(BASE_DIR, "data50", "tf_" + s)
        os.makedirs(outs, exist_ok=True)
        ofs = [tf.python_io.TFRecordWriter(
            outs + "/{}.tfr".format(i)) for i in range(5)]

        with bz2.BZ2File(ins, "rb") as df:
            while True:
                try:
                    line = pickle.load(df)

                    # Write tf records
                    tf_record = wrap_image(line["i"].astype(np.float32),
                                           int(line["l"]))
                    tf_writer = ofs[np.random.randint(0, 5)]
                    tf_writer.write(tf_record.SerializeToString())
                except EOFError:
                    break

        for i in range(5): ofs[i].close()


def wrap_image(image, label_score):
    """
    This function is used to wrap an image into a tfrecord.
    :param image: a standard image
    :param label_score: label of the image
    :return:
    """
    example = tf.train.Example(features=tf.train.Features(feature={
        "i": bytes_feature(image.tostring()),
        "l": int64_feature(label_score)}))

    return example


if __name__ == '__main__':
    main()
    #convert_2tf()











