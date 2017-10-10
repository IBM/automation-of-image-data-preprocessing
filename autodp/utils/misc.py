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
    input("Press the <ENTER> key to continue...")


def get_class(class_name):
    """
    Get a class module from its name.
    :param class_name:
    :return: class module
    """
    sub_mods = class_name.split(sep=".")
    module = __import__(".".join(sub_mods[:-1]), fromlist=sub_mods[-1])
    class_module = getattr(module, sub_mods[-1])

    return class_module


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.
    :param X: ND-Array. Probably should be floats.
    :param theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    :param axis (optional): axis to compute values along. Default is the
        first non-singleton axis.
    :return: an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p





























