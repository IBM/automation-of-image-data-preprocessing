import tensorflow as tf
from tensorflow.python.platform import gfile


def declare_variable(name, shape, initializer):
    """Helper to create a variable."""
    return tf.get_variable(name=name, shape=shape, initializer=initializer, dtype=tf.float32)


def declare_variable_weight_decay(name, shape, initializer, wd):
    """Helper to create an initialized variable with weight decay."""
    var = declare_variable(name=name, shape=shape, initializer=initializer)
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd)
    tf.add_to_collection(tf.GraphKeys.LOSSES, weight_decay)
    return var


def copy_network(tf_vars, coef=1):
    """Copy weight parameters from one network to another resembled network."""
    num_vars = len(tf_vars)
    op = []
    for (idx, var) in enumerate(tf_vars[0:num_vars//2]):
        op.append(tf_vars[idx + num_vars//2].assign((var.value()*coef) + ((1-coef)*tf_vars[idx + num_vars//2].value())))
    return op


def update_network(sess, ops):
    """Update weight parameters after copying them from another network."""
    sess.run(ops)


def create_graph(model):
    """"Creates a graph from saved GraphDef file and returns a saver."""
    with tf.Session() as sess:
        with gfile.FastGFile(model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
    return sess.graph
