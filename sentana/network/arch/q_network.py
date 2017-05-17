"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from sentana.network.arch.base_arch import BaseArch
from sentana.utils.tf_utils import declare_variable
from sentana.utils.tf_utils import declare_variable_weight_decay
from sentana.utils.tf_utils import activation_summary
from sentana.utils.misc import xavier_init
from sentana.config.cf_container import Config as cf


@BaseArch.register
class QNetwork(BaseArch):
    """
    This class implements a simple Q network.
    """
    def __init__(self, instance):
        """
        Initialize by storing the input instance.
        :param instance:
        """
        super().__init__(instance)

    def build_arch(self):
        """
        Build the concrete Q network.
        :return:
        """
        initializer = tf.truncated_normal_initializer(stddev=0.01,
                                                      dtype=tf.float32)

        with tf.variable_scope("conv1") as scope:
            kernel = declare_variable_weight_decay(name="kernel",
                                                   shape=[11, 11, 3, 96],
                                                   initializer=initializer,
                                                   wd=0.0)
            conv = tf.nn.conv2d(self._instance, kernel, [1, 4, 4, 1],
                                padding="VALID")
            bias = declare_variable(name="bias", shape=[96],
                                    initializer=initializer)
            pre_activation = tf.nn.bias_add(conv, bias)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            activation_summary(conv1)

        pool1 = tf.reduce_max(conv1, reduction_indices=[3],
                              keep_dims=True, name="pool1")
        norm1 = tf.nn.lrn(input=pool1, depth_radius=4, bias=1.0,
                          alpha=0.001 / 9.0, beta=0.75, name="norm1")

        with tf.variable_scope("conv2") as scope:
            kernel = declare_variable_weight_decay(name="kernel",
                                                   shape=[5, 5, 1, 256],
                                                   initializer=initializer,
                                                   wd=0.0)
            conv = tf.nn.conv2d(norm1, kernel, [1, 2, 2, 1], padding="VALID")
            bias = declare_variable(name="bias", shape=[256],
                                    initializer=initializer)
            pre_activation = tf.nn.bias_add(conv, bias)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            activation_summary(conv2)

        pool2 = tf.reduce_max(conv2, reduction_indices=[3],
                              keep_dims=True, name="pool2")
        norm2 = tf.nn.lrn(input=pool2, depth_radius=4, bias=1.0,
                          alpha=0.001 / 9.0, beta=0.75, name="norm2")

        with tf.variable_scope("fc1") as scope:
            shape = norm2.get_shape().as_list()
            dim = shape[1] * shape[2]
            reshape = tf.reshape(norm2, [-1, dim])
            weights = declare_variable_weight_decay(name="weight",
                                                    shape=[dim, 512],
                                                    initializer=initializer,
                                                    wd=0.0)
            bias = declare_variable(name="bias", shape=[512],
                                    initializer=initializer)
            fc1 = tf.nn.relu(tf.matmul(reshape, weights) + bias,
                             name=scope.name)
            activation_summary(fc1)

        with tf.variable_scope("fc2") as scope:
            weights = declare_variable_weight_decay(name="weight",
                                                    shape=[512, 512],
                                                    initializer=initializer,
                                                    wd=0.0)
            bias = declare_variable(name="bias", shape=[512],
                                    initializer=initializer)
            fc2 = tf.nn.relu(tf.matmul(fc1, weights) + bias, name=scope.name)
            activation_summary(fc2)

        # Implement the Dual-Q network
        # Split into separate advantage and value
        pre_adv, pre_val = tf.split(fc2, 2, 1)
        w_adv = declare_variable_weight_decay(name="w_adv", wd=0.0,
                                              shape=[256, cf.num_action],
                                              initializer=xavier_init)
        w_val = declare_variable_weight_decay(name="w_val", shape=[256, 1],
                                              initializer=xavier_init, wd=0.0)
        advantage = tf.matmul(pre_adv, w_adv)
        value = tf.matmul(pre_val, w_val)

        # Combine them together to get final Q value
        q_out = value + tf.sub(advantage, tf.reduce_mean(advantage, axis=1,
                                                         keep_dims=True))

        return (q_out, tf.arg_max(q_out, 1))
