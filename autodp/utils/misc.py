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
