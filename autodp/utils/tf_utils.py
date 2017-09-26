"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf
from tensorflow.python.platform import gfile


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
        if shape is None:
            var = tf.get_variable(name=name, dtype=tf.float32,
                                  initializer=initializer)
        else:
            var = tf.get_variable(name=name, shape=shape,
                                  initializer=initializer,
                                  dtype=tf.float32)

    return var


def declare_variable_weight_decay(name, initializer, wd, shape=None, dev="cpu"):
    """
    Helper to create an initialized variable with weight decay.
    :param name:
    :param shape:
    :param initializer:
    :param wd:
    :param dev:
    :return:
    """
    var = declare_variable(name=name, shape=shape,
                           initializer=initializer,
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


def int64_feature(value):
    """
    Encode a feature to int64.
    :param value:
    :return:
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    """
    Encode a feature to bytes.
    :param value:
    :return:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _to_tensor(x, dtype):
    """Convert the input x to a tensor of type dtype.
    :param x: An object to be converted (numpy array, list, tensors)
    :param dtype: The destination type
    :return: A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)

    return x


def relu(x, alpha=0, max_value=None):
    """ Leaky rectified linear unit, with default values, it returns
    element-wise max(x, 0).
    :param x: A tensor or variable
    :param alpha: Slope of negative section (should be smaller than -1)
    :param max_value: Saturation threshold
    :return: A tensor
    """
    if alpha != 0.:
        negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        max_value = _to_tensor(max_value, x.dtype.base_dtype)
        zero = _to_tensor(0., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, max_value)
    if alpha != 0.:
        alpha = _to_tensor(alpha, x.dtype.base_dtype)
        x -= alpha * negative_part

    return x


def copy_network(sess, tf_vars, coef=1):
    """
    Copy weight parameters from one network to another resembled network.
    :param sess:
    :param tf_vars:
    :param coef:
    :return:
    """
    num_vars = len(tf_vars)
    for (idx, var) in enumerate(tf_vars[0:num_vars//2]):
        assign_op = tf_vars[idx + num_vars//2].assign((var.value()*coef) + (
            (1-coef)*tf_vars[idx + num_vars//2].value()))
        sess.run(assign_op)


def create_graph(model):
    """"Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.Session() as sess:
        with gfile.FastGFile(model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
    return sess.graph

















