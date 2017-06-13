"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os

import tensorflow as tf
import cv2 as cv
import numpy as np
import multiprocessing

from sentana.utils.tf_utils import bytes_feature
from sentana.utils.tf_utils import int64_feature


IM_DIR = "/Users/minhtn/ibm/origin_data/sentana/you_data/images.bk"
TFRECORD = "/Users/minhtn/ibm/projects/sentana/data/you_data2"
IM_SIZE = 256


def process_image(im_path):
    # Load image
    im = cv.imread(im_path)

    # Resize image
    ml = np.min(im.shape[:-1])
    resize_im = cv.resize(im, (int(im.shape[1] / ml * IM_SIZE),
                               int(im.shape[0] / ml * IM_SIZE)))

    # Crop image
    dim0 = int((resize_im.shape[0] - IM_SIZE) / 2)
    dim1 = int((resize_im.shape[1] - IM_SIZE) / 2)
    crop_im = resize_im[dim0: dim0+IM_SIZE, dim1: dim1+IM_SIZE, :]

    # Normalize image
    #dst = np.zeros(shape=crop_im.shape)
    #norm_im = cv.normalize(crop_im, dst, alpha=0, beta=1,
    #                       norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    norm_im = np.float32(crop_im / 255.0)

    # Store image
    #names = im_path.split(sep="/")
    #cv.imwrite("/" + os.path.join(*names[:-1], "crop_" + names[-1]), norm_im)

    return norm_im


def wrap_image(image, label_index):
    """
    This function is used to wrap an image into a tfrecord.
    :param image: a standard image
    :param label_index: label of the image
    :return:
    """
    example = tf.train.Example(features=tf.train.Features(feature={
        "img": bytes_feature(image.tostring()),
        "label": int64_feature(label_index)}))

    return example


def worker(label, label_index):
    """
    This function is run in parallel.
    :param label:
    :return:
    """
    tfrecord_writer = tf.python_io.TFRecordWriter(os.path.join(
        TFRECORD, label + ".tfrecord"))

    # Process images for each label
    files = os.listdir(os.path.join(IM_DIR, label))
    files = [f for f in files if not f.startswith(".")]
    for file in files:
        sd_image = process_image(os.path.join(IM_DIR, label, file))
        tf_record = wrap_image(sd_image, label_index)
        tfrecord_writer.write(tf_record.SerializeToString())

    tfrecord_writer.close()

    return


def parallel_convert():
    """
    Convert images into tfrecords.
    :return:
    """
    # Create label indices
    labels = os.listdir(IM_DIR)
    labels = [l for l in labels if not l.startswith(".")]
    label_dict = {l: i for (i, l) in enumerate(labels)}

    # Parallel processing images
    jobs = []
    for label in labels:
        p = multiprocessing.Process(target=worker, args=(label,
                                                         label_dict[label]))
        jobs.append(p)
        p.start()
        print(p.pid)


if __name__ == "__main__":
    parallel_convert()

