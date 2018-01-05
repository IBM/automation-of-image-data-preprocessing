"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os

import numpy as np
from PIL import Image
import pickle
import bz2
import cv2 as cv


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


def prep_images(data, label, out_file):
    """
    Preprocess images. Reads images in paths, and writes to out_dir.
    """
    writer = bz2.BZ2File(out_file, "wb")
    for i in range(len(label)):
        img_flat = data[i]
        img_R = img_flat[0:1024].reshape((32, 32))
        img_G = img_flat[1024:2048].reshape((32, 32))
        img_B = img_flat[2048:3072].reshape((32, 32))
        img = np.dstack((img_R, img_G, img_B))
        img_nrm = np.asarray(norm_image(Image.fromarray(img)))
        line = {"i": img_nrm, "l": np.int32(label[i])}
        pickle.dump(line, writer, pickle.HIGHEST_PROTOCOL)

    writer.close()


def dist_images(data, label, out_file):
    """
    Preprocess images. Reads images in paths, and writes to out_dir.
    """
    writer = bz2.BZ2File(out_file, "wb")
    for i in range(len(label)):
        img_flat = data[i]
        img_R = img_flat[0:1024].reshape((32, 32))
        img_G = img_flat[1024:2048].reshape((32, 32))
        img_B = img_flat[2048:3072].reshape((32, 32))
        img = process_img(np.dstack((img_R, img_G, img_B)))
        img_nrm = np.asarray(norm_image(Image.fromarray(img)))
        line = {"i": img_nrm, "l": np.int32(label[i])}
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

    return np.reshape(img, [32, 32, 3])


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


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict


def preprocess():
    # Change these paths to match the location of the CUB dataset on your machine
    dataset_dir = "/Users/minhtn/ibm/projects/autodp/storage/inputs/cifar10/cifar-10-batches-py"

    ds1 = unpickle(os.path.join(dataset_dir, "data_batch_1"))
    ds2 = unpickle(os.path.join(dataset_dir, "data_batch_2"))
    ds3 = unpickle(os.path.join(dataset_dir, "data_batch_3"))
    ds4 = unpickle(os.path.join(dataset_dir, "data_batch_4"))
    ds5 = unpickle(os.path.join(dataset_dir, "data_batch_5"))

    train_data = np.vstack([ds1[b'data'], ds2[b'data'], ds3[b'data'], ds4[b'data'], ds5[b'data'][:9000]])
    train_label = ds1[b'labels'] + ds2[b'labels'] + ds3[b'labels'] + ds4[b'labels'] + ds5[b'labels'][:9000]
    valid_data = ds5[b'data'][9000:]
    valid_label = ds5[b'labels'][9000:]
    test_data = unpickle(os.path.join(dataset_dir, "test_batch"))[b'data']
    test_label = unpickle(os.path.join(dataset_dir, "test_batch"))[b'labels']

    # Store datasets
    prep_images(train_data, train_label, "/Users/minhtn/ibm/projects/autodp/storage/inputs/cifar10/train.bz2")
    prep_images(valid_data, valid_label, "/Users/minhtn/ibm/projects/autodp/storage/inputs/cifar10/valid.bz2")
    prep_images(test_data, test_label, "/Users/minhtn/ibm/projects/autodp/storage/inputs/cifar10/test.bz2")
    dist_images(test_data, test_label, "/Users/minhtn/ibm/projects/autodp/storage/inputs/cifar10/dist_test.bz2")


if __name__ == "__main__":
    preprocess()










