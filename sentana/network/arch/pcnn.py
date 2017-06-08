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
        initializer = tf.truncated_normal_initializer(dtype=tf.float32,
                                                      stddev=0.1)

        with tf.variable_scope("conv1") as scope:
            kernel = declare_variable_weight_decay(initializer=initializer,
                                                   name="kernel", wd=0.0,
                                                   shape=[11, 11, 3, 6])
            conv = tf.nn.conv2d(self._instance, kernel, [1, 4, 4, 1], "VALID")
            bias = declare_variable(name="bias", shape=[6],
                                    initializer=initializer)
            pre_activation = tf.nn.bias_add(conv, bias)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            activation_summary(conv1)

        pool1 = tf.reduce_max(conv1, reduction_indices=[3],
                              keep_dims=True, name="pool1")
        #norm1 = tf.nn.lrn(input=pool1, name="norm1")

        with tf.variable_scope("conv2") as scope:
            kernel = declare_variable_weight_decay(initializer=initializer,
                                                   name="kernel", wd=0.0,
                                                   shape=[5, 5, 1, 5])
            conv = tf.nn.conv2d(pool1, kernel, [1, 2, 2, 1], padding="VALID")
            bias = declare_variable(name="bias", shape=[5],
                                    initializer=initializer)
            pre_activation = tf.nn.bias_add(conv, bias)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            activation_summary(conv2)

        pool2 = tf.reduce_max(conv2, reduction_indices=[3],
                              keep_dims=True, name="pool2")
        #norm2 = tf.nn.lrn(input=pool2, name="norm2")

        with tf.variable_scope("fc") as scope:
            shape = pool2.get_shape().as_list()
            dim = shape[1] * shape[2]
            rsh = tf.reshape(pool2, [-1, dim])
            weights = declare_variable_weight_decay(name="weight",
                initializer=xavier_init(dim, cf.num_class), wd=0.0)
            bias = declare_variable(name="bias", shape=[cf.num_class],
                                    initializer=initializer)
            fc = tf.matmul(rsh, weights) + bias
            activation_summary(fc)

        return tf.nn.softmax(fc)
