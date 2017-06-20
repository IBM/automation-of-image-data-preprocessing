"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from sentana.network.arch.base_arch import BaseArch
from sentana.utils.tf_utils import declare_variable
from sentana.utils.tf_utils import declare_variable_weight_decay
from sentana.utils.tf_utils import activation_summary
from sentana.config.cf_container import Config as cf
from sentana.utils.misc import xavier_init


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def relu(x, alpha=0, max_value=None):
    """Rectified linear unit.
    With default values, it returns element-wise `max(x, 0)`.
    # Arguments
        x: A tensor or variable.
        alpha: A scalar, slope of negative section (default=`0.`).
        max_value: Saturation threshold.
    # Returns
        A tensor.
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


@BaseArch.register
class PCNN(BaseArch):
    """
    This class implements the network architecture Progressive CNN presented
     in the paper entitled "Robust Image Sentiment Analysis Using Progressively
     Trained and Domain Transferred Deep Networks".
    """
    def __init__(self, instance):
        """
        Initialize by storing the input instance.
        :param instance:
        """
        super().__init__(instance)

    def build_arch(self):
        """
        Build the concrete PCNN architecture.
        :return:
        """
        initializer1 = tf.truncated_normal_initializer(dtype=tf.float32,
                                                      stddev=0.1)
        initializer2 = tf.truncated_normal_initializer(dtype=tf.float32,
                                                       stddev=0.01)
        initializer1 = tf.uniform_unit_scaling_initializer()
        initializer2 = tf.uniform_unit_scaling_initializer(factor=0.001)
        initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("conv1") as scope:
            kernel1 = declare_variable_weight_decay(initializer=initializer1,
                                                   name="kernel", wd=0.0,
                                                   shape=[4, 4, 3, 10])
            conv = tf.nn.conv2d(tf.cast(self._instance, tf.float32), kernel1, [1, 2, 2, 1], "VALID")
            bias1 = declare_variable(name="bias", shape=[10],
                                    initializer=initializer2)
            pre_activation1 = tf.nn.bias_add(conv, bias1)
            #conv1 = tf.nn.relu(pre_activation, name=scope.name)
            conv1 = relu(pre_activation1)
            activation_summary(conv1)

        pool1 = tf.reduce_max(conv1, reduction_indices=[3],
                              keep_dims=True, name="pool1")
        #norm1 = tf.nn.lrn(input=pool1, name="norm1")

        with tf.variable_scope("conv2") as scope:
            kernel = declare_variable_weight_decay(initializer=initializer1,
                                                   name="kernel", wd=0.0,
                                                   shape=[3, 3, 1, 5])
            conv = tf.nn.conv2d(pool1, kernel, [1, 2, 2, 1], padding="VALID")
            bias = declare_variable(name="bias", shape=[5],
                                    initializer=initializer2)

            pre_activation2 = tf.nn.bias_add(conv, bias)
            #conv2 = tf.nn.relu(pre_activation, name=scope.name)
            conv2 = relu(pre_activation2)
            activation_summary(conv2)

        pool2 = tf.reduce_max(conv2, reduction_indices=[3],
                              keep_dims=True, name="pool2")
        #norm2 = tf.nn.lrn(input=pool2, name="norm2")

        with tf.variable_scope("conv3") as scope:
            kernel = declare_variable_weight_decay(initializer=initializer,
                                                   name="kernel", wd=0.0,
                                                   shape=[3, 3, 1, 1])
            conv = tf.nn.conv2d(pool2, kernel, [1, 2, 2, 1], padding="VALID")
            bias = declare_variable(name="bias", shape=[1],
                                    initializer=initializer)
            pre_activation = tf.nn.bias_add(conv, bias)
            conv3 = tf.nn.relu(pre_activation, name=scope.name)
            activation_summary(conv3)

        pool3 = tf.reduce_max(conv3, reduction_indices=[3],
                               keep_dims=True, name="pool3")
        #norm2 = tf.nn.lrn(input=pool2, name="norm2")

        with tf.variable_scope("conv4") as scope:
            kernel = declare_variable_weight_decay(initializer=initializer,
                                                   name="kernel", wd=0.0,
                                                   shape=[4, 4, 1, 1])
            conv = tf.nn.conv2d(pool3, kernel, [1, 2, 2, 1], padding="VALID")
            bias = declare_variable(name="bias", shape=[1],
                                    initializer=initializer)
            pre_activation = tf.nn.bias_add(conv, bias)
            conv4 = tf.nn.relu(pre_activation, name=scope.name)
            activation_summary(conv4)

        pool4 = tf.reduce_max(conv4, reduction_indices=[3],
                              keep_dims=True, name="pool4")
        #norm2 = tf.nn.lrn(input=pool2, name="norm2")


        # with tf.variable_scope("fc0") as scope:
        #     shape = pool2.get_shape().as_list()
        #     dim = shape[1] * shape[2]
        #     rsh = tf.reshape(pool2, [-1, dim])
        #     weights = declare_variable_weight_decay(name="weight",
        #         initializer=xavier_init(dim, 256), wd=0.0)
        #     bias = declare_variable(name="bias", shape=[256],
        #                             initializer=initializer)
        #     fc0 = tf.nn.relu(tf.matmul(rsh, weights) + bias, name=scope.name)
        #     activation_summary(fc0)

        with tf.variable_scope("fc1") as scope:
            shape = pool2.get_shape().as_list()
            dim = shape[1] * shape[2]
            rsh = tf.reshape(pool2, [-1, dim])
            weights = declare_variable_weight_decay(name="weight", shape=[dim, 128],
                initializer=initializer, wd=0.0)
            bias = declare_variable(name="bias", shape=[128],
                                    initializer=initializer1)

            fc1 = relu(tf.matmul(rsh, weights) + bias)
            #fc1 = tf.matmul(rsh, weights) + bias
            activation_summary(fc1)

        with tf.variable_scope("fc2") as scope:
            weights = declare_variable_weight_decay(name="weight",
                                                    shape=[128, 64],
                                                    initializer=initializer,
                                                    wd=0.0)
            bias = declare_variable(name="bias", shape=[64],
                                    initializer=initializer1)
            fc2 = relu(tf.matmul(fc1, weights) + bias)
            activation_summary(fc2)

        with tf.variable_scope("fc3") as scope:
            weights = declare_variable_weight_decay(name="weight",
                                                    shape=[64, 32],
                                                    initializer=initializer,
                                                    wd=0.0)
            bias = declare_variable(name="bias", shape=[32],
                                    initializer=initializer1)
            fc3 = relu(tf.matmul(fc2, weights) + bias)
            activation_summary(fc3)

        with tf.variable_scope("fc4") as scope:
            weights = declare_variable_weight_decay(name="weight",
                                                    shape=[32, 8],
                                                    initializer=initializer,
                                                    wd=0.0)
            bias = declare_variable(name="bias", shape=[8],
                                    initializer=initializer1)
            fc4 = relu(tf.add(tf.matmul(fc3, weights), bias, name=scope.name))
            activation_summary(fc4)

        with tf.variable_scope("fc5") as scope:
            weights = declare_variable_weight_decay(name="weight",
                                                    shape=[8, cf.num_class],
                                                    initializer=initializer,
                                                    wd=0.0)
            bias = declare_variable(name="bias", shape=[2],
                                    initializer=initializer1)
            fc5 = (tf.add(tf.matmul(fc4, weights), bias, name=scope.name))
            activation_summary(fc5)



        return (fc5, [kernel1, bias1, conv1, pool1])
