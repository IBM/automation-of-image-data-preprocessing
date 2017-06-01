"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os

import cv2 as cv
import numpy as np
import json
import multiprocessing
import tensorflow as tf

from sentana.utils.tf_utils import bytes_feature
from sentana.utils.tf_utils import int64_feature


IM_DIR = "/dccstor/sentana/sentana/cc_data"
OUT_DIR = "/dccstor/sentana/sentana/model/cc_data"
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
    labels = [l for l in labels if not l.startswith(".")]
    jobs = []
    lpp = int(len(labels) / 4)
    for i in range(4):
        if i < 3:
            p = multiprocessing.Process(
                target=worker, args=(labels[i*lpp:(i+1)*lpp], label_dict))
        else:
            p = multiprocessing.Process(target=worker,
                                        args=(labels[i*lpp:], label_dict))

        jobs.append(p)
        p.start()
        print(p.pid)


def get_labels(filename):
    """
    Parse the visual sentiment ontology file to get sentiment values.
    :param filename:
    :return:
    """
    with open(filename) as f:
        content = f.readlines()[3:]
        print (content[0])

    label_dict = {}
    anp_dict = {}
    for line in content:
        if line.startswith("\t"):
            label_dict[line.split(" ")[0][1:]] = float(line.split(" ")[2][:-1])
            anp_dict[line.split(" ")[0][1:]] = float(line.split(" ")[2][:-1])
        else:
            label_dict[line.split(" ")[1]] = float(line.split(" ")[3][:-1])

    return label_dict, anp_dict


def worker(labels, label_dict):
    """
    This function is run in parallel.
    :param labels:
    :param label_dict:
    :return:
    """
    for label in labels:
        # Writers for tfrecords
        tf_writer_train = tf.python_io.TFRecordWriter(os.path.join(
            OUT_DIR, "tf_train", label + ".tftrain"))
        tf_writer_valid = tf.python_io.TFRecordWriter(os.path.join(
            OUT_DIR, "tf_valid", label + ".tfvalid"))
        tf_writer_test = tf.python_io.TFRecordWriter(os.path.join(
            OUT_DIR, "tf_test", label + ".tftest"))

        # Writer for ril
        ril_writer_train = open(os.path.join(OUT_DIR, "ril_train",
                                             label + ".riltrain"), "w")
        ril_writer_valid = open(os.path.join(OUT_DIR, "ril_valid",
                                             label + ".rilvalid"), "w")
        ril_writer_test = open(os.path.join(OUT_DIR, "ril_test",
                                             label + ".riltest"), "w")

        # Process images for each label
        files = os.listdir(os.path.join(IM_DIR, label))
        files = [os.path.join(IM_DIR, label, f) for f in files
                 if not f.startswith(".")]
        label_score = 1 if label_dict[label] >= 0 else 0

        worker_assist(files[:int(0.8*len(files))], label_score,
                      tf_writer_train, ril_writer_train)
        worker_assist(files[int(0.8*len(files)):int(0.9*len(files))],
                      label_score, tf_writer_valid, ril_writer_valid)
        worker_assist(files[int(0.9*len(files)):], label_score,
                      tf_writer_test, ril_writer_test)

        tf_writer_train.close()
        tf_writer_valid.close()
        tf_writer_test.close()
        ril_writer_train.close()
        ril_writer_valid.close()
        ril_writer_test.close()

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
        line = {"img": np.array(sd_image).tolist(), "label": label_score}
        json.dump(line, ril_writer)
        ril_writer.write("\n")


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
        "image": bytes_feature(image.tostring()),
        "label": int64_feature(label_score)}))

    return example


if __name__ == "__main__":
    process()

    #shuf huge_file.txt -o shuffled_lines_huge_file.txt