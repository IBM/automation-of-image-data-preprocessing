"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os

import cv2 as cv
import numpy as np
import multiprocessing
import tensorflow as tf
import bz2
import pickle
import random

from sentana.utils.tf_utils import bytes_feature
from sentana.utils.tf_utils import int64_feature


IM_DIR = "/dccstor/sentana/sentana/cc_data"
OUT_DIR = "/dccstor/sentana/sentana/model"
LABEL = "/dccstor/sentana/sentana/Dataset_Vso/ANPs/3244ANPs.txt"
IM_SIZE = 256


def process():
    """
    Main function to process images.
    :return:
    """
    # Create label indices
    _, label_dict = get_labels(LABEL)

    # Process images for each label, do in parallel
    #labels = list(label_dict.keys())
    labels = os.listdir(IM_DIR)
    labels = [l for l in labels if not l.startswith(".")
              and l in label_dict.keys()]
    random.shuffle(labels)

    jobs = []
    lpp = 100
    for i in range(4):
        p = multiprocessing.Process(target=worker,
            args=(labels[i*lpp:(i+1)*lpp], label_dict, "cc_data" + str(i)))

        jobs.append(p)
        p.start()
        p.join()
        print(p.pid)


def merge(labels, dataset, part):
    """
    Randomly merge bz2 files together.
    :param labels:
    :param dataset:
    :param part:
    :return:
    """
    files = [os.path.join(OUT_DIR, dataset, part, l + ".bz2") for l in labels]
    readers = [bz2.BZ2File(file, "rb") for file in files]
    writer = bz2.BZ2File(os.path.join(OUT_DIR, dataset, part,
                                      part + ".bz2"), "wb")

    while len(readers) > 0:
        random.shuffle(readers)
        for (i, r) in enumerate(readers):
            try:
                line = pickle.load(r)
                pickle.dump(line, writer)
            except EOFError:
                r.close()
                os.remove(r.name)
                del readers[i]
                break

            if np.random.rand() < 0.3: break

    writer.close()


def get_labels(filename):
    """
    Parse the visual sentiment ontology file to get sentiment values.
    :param filename:
    :return:
    """
    with open(filename) as f:
        content = f.readlines()[3:]

    label_dict = {}
    anp_dict = {}
    for line in content:
        if line.startswith("\t"):
            label_dict[line.split(" ")[0][1:]] = float(line.split(" ")[2][:-1])
            anp_dict[line.split(" ")[0][1:]] = float(line.split(" ")[2][:-1])
        else:
            label_dict[line.split(" ")[1]] = float(line.split(" ")[3][:-1])

    return label_dict, anp_dict


def worker(labels, label_dict, dataset):
    """
    This function is run in parallel.
    :param labels:
    :param label_dict:
    :param dataset:
    :return:
    """
    for label in labels:
        # Writers for tfrecords
        tf_writer_train = tf.python_io.TFRecordWriter(os.path.join(
            OUT_DIR, dataset, "tf_train", label + ".tftrain"))
        tf_writer_valid = tf.python_io.TFRecordWriter(os.path.join(
            OUT_DIR, dataset, "tf_valid", label + ".tfvalid"))
        tf_writer_test = tf.python_io.TFRecordWriter(os.path.join(
            OUT_DIR, dataset, "tf_test", label + ".tftest"))

        # Writer for ril
        ril_writer_train = bz2.BZ2File(os.path.join(OUT_DIR, dataset, "train",
                                                    label + ".bz2"), "wb")
        ril_writer_valid = bz2.BZ2File(os.path.join(OUT_DIR, dataset, "valid",
                                                    label + ".bz2"), "wb")
        ril_writer_test = bz2.BZ2File(os.path.join(OUT_DIR, dataset, "test",
                                                   label + ".bz2"), "wb")

        # Process images for each label
        files = os.listdir(os.path.join(IM_DIR, label))
        files = [os.path.join(IM_DIR, label, f) for f in files
            if not f.startswith(".") and os.path.getsize(
                os.path.join(IM_DIR, label, f)) > 3051]
        label_score = 1 if label_dict[label] >= 0 else 0

        worker_assist(files[:10], label_score, tf_writer=tf_writer_train,
                      ril_writer=ril_writer_train)
        worker_assist(files[10:12], label_score, tf_writer=tf_writer_valid,
                      ril_writer=ril_writer_valid)
        worker_assist(files[12:14], label_score, tf_writer=tf_writer_test,
                      ril_writer=ril_writer_test)

        tf_writer_train.close()
        tf_writer_valid.close()
        tf_writer_test.close()
        ril_writer_train.close()
        ril_writer_valid.close()
        ril_writer_test.close()

    merge(labels, dataset, "train")
    merge(labels, dataset, "valid")
    merge(labels, dataset, "test")

    return


def worker_assist(files, label_score, tf_writer, ril_writer):
    """
    Process each subset of images for each label.
    :param files:
    :param label_score:
    :param tf_writer:
    :param ril_writer:
    :return:
    """
    for file in files:
        if os.path.getsize(file) <= 2051:
            print("Skip file: " + file)
            continue

        sd_image = process_image(file)

        # Write tf records
        tf_record = wrap_image(sd_image, label_score)
        tf_writer.write(tf_record.SerializeToString())

        # Write ril records
        line = {"img": sd_image, "label": label_score}
        pickle.dump(line, ril_writer, pickle.HIGHEST_PROTOCOL)


def process_image(im_path):
    """
    Standardize an image.
    :param im_path:
    :return:
    """
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
    dst = np.zeros(shape=crop_im.shape)
    norm_im = cv.normalize(crop_im, dst, alpha=0, beta=1,
                           norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    # Store image
    #names = im_path.split(sep="/")
    #cv.imwrite("/" + os.path.join(*names[:-1], "crop_" + names[-1]), norm_im)

    return norm_im


def wrap_image(image, label_score):
    """
    This function is used to wrap an image into a tfrecord.
    :param image: a standard image
    :param label_score: label of the image
    :return:
    """
    example = tf.train.Example(features=tf.train.Features(feature={
        "img": bytes_feature(image.tostring()),
        "label": int64_feature(label_score)}))

    return example


if __name__ == "__main__":
    process()

    #shuf huge_file.txt -o shuffled_lines_huge_file.txt