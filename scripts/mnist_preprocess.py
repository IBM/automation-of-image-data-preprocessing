"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os
import struct
import numpy as np
import pickle
import bz2
from PIL import Image
import cv2 as cv


DIR = "/Users/minhtn/ibm/projects/autodp/storage/inputs/mnist"


def read(dataset="training", path="."):
    """
    Python function for importing the MNIST data set.
    """
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)


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


def produce_data():
    # Writers
    train_writer = bz2.BZ2File(os.path.join(DIR, "train.bz2"), "wb")
    valid_writer = bz2.BZ2File(os.path.join(DIR, "valid.bz2"), "wb")
    test_writer = bz2.BZ2File(os.path.join(DIR, "test.bz2"), "wb")
    dist_test_writer = bz2.BZ2File(os.path.join(DIR, "dist_test.bz2"), "wb")

    # Data sets
    train_valid = list(read("training", DIR))
    train = train_valid[:59000]
    valid = train_valid[59000:]
    test = list(read("testing", DIR))

    # Process train, valid, test
    process1(train, train_writer)
    process1(valid, valid_writer)
    process1(test, test_writer)
    process(test, dist_test_writer)

    train_writer.close()
    valid_writer.close()
    test_writer.close()
    dist_test_writer.close()


def process1(ds, writer):
    # Process each dataset
    for (l, i) in ds:
        i = np.reshape(i, [28, 28, 1])
        line = {"i": i, "l": np.int32(l)}
        pickle.dump(line, writer, pickle.HIGHEST_PROTOCOL)


def process(ds, writer):
    # Process each dataset
    for (l, i) in ds:
        i = process_img(i)
        line = {"i": i, "l": np.int32(l)}
        pickle.dump(line, writer, pickle.HIGHEST_PROTOCOL)


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

    return np.reshape(img, [28, 28, 1])


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


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


if __name__ == "__main__":
    produce_data()








