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
    """Save object with pickle."""
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def from_pickle(filename):
    """Load object with pickle."""
    file = open(filename, 'rb')
    return pickle.load(file)


def clear_model_dir(mdir):
    """Helper to clear model directory."""
    if os.path.isdir(mdir):
        shutil.rmtree(mdir)
    os.makedirs(mdir)


def xavier_init(fan_in, fan_out, const=1, dtype=tf.float32):
    """Xavier initialization."""
    low = -const * np.sqrt(3.0 / (fan_in + fan_out))
    high = const * np.sqrt(3.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=dtype)


def get_class(class_name):
    """Get a class module from its name."""
    sub_mods = class_name.split(sep=".")
    module = __import__(".".join(sub_mods[:-1]), fromlist=sub_mods[-1])
    class_module = getattr(module, sub_mods[-1])
    return class_module


def softmax(X, theta=1.0, axis=None):
    """Compute the softmax of each element along an axis of X."""
    y = np.atleast_2d(X)

    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)

    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum

    if len(X.shape) == 1: p = p.flatten()
    return p
