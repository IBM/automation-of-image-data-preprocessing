"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf


def declare_variable(name, shape, initializer, dev="cpu"):
    """
    Helper to create a variable stored on CPU/GPU memory.
    :param name:
    :param shape:
    :param initializer:
    :param cpu:
    :return: Variable Tensor
    """
    device = "/{}:0".format(dev)
    with tf.device(device):
        var = tf.get_variable(name=name, shape=shape, initializer=initializer,
                              dtype=tf.float32)

    return var


def declare_variable_weight_decay(name, shape, initializer, wd, dev="cpu"):
    """
    Helper to create an initialized variable with weight decay.
    :param name:
    :param shape:
    :param initializer:
    :param wd:
    :param dev:
    :return:
    """
    var = declare_variable(name=name, shape=shape, initializer=initializer,
                           dev=dev)
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd)
    tf.add_to_collection(tf.GraphKeys.LOSSES, weight_decay)

    return var


def activation_summary(x):
    """
    Helper to create summaries for activations.
    :param x:
    :return:
    """
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + "/activations", x)
    tf.summary.scalar(tensor_name + "/sparsity", tf.nn.zero_fraction(x))
