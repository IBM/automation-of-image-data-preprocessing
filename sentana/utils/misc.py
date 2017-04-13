"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os

import shutil
import numpy as np
import tensorflow as tf


def clear_model_dir(mdir):
    """
    Helper to clear model directory.
    :param mdir:
    :return:
    """
    if os.path.isdir(mdir):
        shutil.rmtree(mdir)

    os.mkdir(mdir)


def xavier_init(fan_in, fan_out, const=1):
    """
    Xavier initialization.
    :param fan_in:
    :param fan_out:
    :param const:
    :return:
    """
    low = -const * np.sqrt(3.0 / (fan_in + fan_out))
    high = const * np.sqrt(3.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)


def weighted_pick(weights):
    """
    Random weight pickup.
    :param weights:
    :return:
    """
    t = np.cumsum(weights)
    s = np.sum(weights)

    return int(np.searchsorted(t, np.random.rand(1) * s))

