"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os

import shutil
import numpy as np
import tensorflow as tf
import pickle


def to_pickle(filename, obj):
    """
    Save object with pickle.
    :param filename:
    :param obj:
    :return:
    """
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def from_pickle(filename):
    """
    Load object with pickle.
    :param filename:
    :return:
    """
    file = open(filename,'rb')

    return pickle.load(file)


def clear_model_dir(mdir):
    """
    Helper to clear model directory.
    :param mdir:
    :return:
    """
    if os.path.isdir(mdir):
        shutil.rmtree(mdir)

    os.makedirs(mdir)


def xavier_init(fan_in, fan_out, const=1, dtype=tf.float32):
    """
    Xavier initialization.
    :param fan_in:
    :param fan_out:
    :param const:
    :return:
    """
    low = -const * np.sqrt(3.0 / (fan_in + fan_out))
    high = const * np.sqrt(3.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low,
                             maxval=high, dtype=dtype)


def weighted_pick(weights):
    """
    Random weight pickup.
    :param weights:
    :return:
    """
    t = np.cumsum(weights)
    s = np.sum(weights)

    return int(np.searchsorted(t, np.random.rand(1) * s))


def pause():
    """
    Pause the program until enter key pressed.
    :return:
    """
    programPause = input("Press the <ENTER> key to continue...")

