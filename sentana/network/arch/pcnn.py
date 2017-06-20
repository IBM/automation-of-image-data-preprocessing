"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from sentana.network.arch.base_arch import BaseArch
from sentana.utils.tf_utils import declare_variable
from sentana.utils.tf_utils import declare_variable_weight_decay
from sentana.config.cf_container import Config as cf
from sentana.utils.tf_utils import relu


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
        kern_init = tf.uniform_unit_scaling_initializer()
        bias_init = tf.uniform_unit_scaling_initializer(factor=0.001)
        xavi_init = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("conv1") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[4, 4, 3, 10])
            conv = tf.nn.conv2d(tf.cast(self._instance, tf.float32),
                                kernel, [1, 2, 2, 1], "VALID")
            bias = declare_variable(name="bias", shape=[10],
                                    initializer=bias_init)
            pool1 = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")

        with tf.variable_scope("conv2") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[3, 3, 1, 5])
            conv = tf.nn.conv2d(pool1, kernel, [1, 2, 2, 1], padding="VALID")
            bias = declare_variable(name="bias", shape=[5],
                                    initializer=bias_init)
            pool2 = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")

        with tf.variable_scope("fc1") as scope:
            shape = pool2.get_shape().as_list()
            dim = shape[1] * shape[2]
            rsh = tf.reshape(pool2, [-1, dim])
            weights = declare_variable_weight_decay(initializer=xavi_init,
                name="weight", shape=[dim, 128], wd=0.0)
            bias = declare_variable(name="bias", shape=[128],
                                    initializer=bias_init)
            fc1 = relu(tf.nn.bias_add(tf.matmul(rsh, weights), bias))

        with tf.variable_scope("fc2") as scope:
            weights = declare_variable_weight_decay(initializer=xavi_init,
                name="weight", shape=[128, cf.num_class], wd=0.0)
            bias = declare_variable(name="bias", shape=[cf.num_class],
                                    initializer=bias_init)
            fc2 = tf.nn.bias_add(tf.matmul(fc1, weights), bias)

        return fc2
