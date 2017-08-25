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
        kern_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.truncated_normal_initializer(stddev=0.01)
        xavi_init = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("conv1") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0008, shape=[4, 4, 1, 20])
            conv = tf.nn.conv2d(tf.cast(self._instance, tf.float32),
                                kernel, [1, 1, 1, 1], "SAME")
            bias = declare_variable(name="bias", shape=[20],
                                    initializer=bias_init)
            norm = tf.nn.lrn(input=relu(tf.nn.bias_add(conv, bias)),
                             name="norm_pool")
            pool1 = tf.nn.max_pool(norm, ksize=[1, 3, 3, 1],
                                   strides=[1, 1, 1, 1], padding="VALID")

        with tf.variable_scope("conv2") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0008, shape=[5, 5, 20, 15])
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[15],
                                    initializer=bias_init)
            #norm = tf.nn.lrn(input=relu(tf.nn.bias_add(conv, bias)),
            #                 name="norm_pool")
            pool2 = tf.nn.max_pool(relu(tf.nn.bias_add(conv, bias)),
                                   ksize=[1, 3, 3, 1],
                                   strides=[1, 1, 1, 1], padding="VALID")

        with tf.variable_scope("fc1") as scope:
            shape = pool2.get_shape().as_list()
            dim = shape[1] * shape[2] * shape[3]
            rsh = tf.reshape(pool2, [-1, dim])
            weights = declare_variable_weight_decay(initializer=xavi_init,
                name="weight", shape=[dim, 256], wd=0.0008)
            bias = declare_variable(name="bias", shape=[256],
                                    initializer=bias_init)
            fc1 = relu(tf.nn.bias_add(tf.matmul(rsh, weights), bias))

        with tf.variable_scope("fc2") as scope:
            weights = declare_variable_weight_decay(initializer=xavi_init,
                name="weight", shape=[256, 32], wd=0.0008)
            bias = declare_variable(name="bias", shape=[32],
                                    initializer=bias_init)
            fc2 = relu(tf.nn.bias_add(tf.matmul(fc1, weights), bias))

        # Implement the Dual-Q network
        # Split into separate advantage and value
        pre_adv, pre_val = tf.split(value=fc2, num_split=2, split_dim=1)
        w_adv = declare_variable_weight_decay(initializer=xavi_init,
            name="w_adv", wd=0.0008, shape=[16, cf.num_action])
        w_val = declare_variable_weight_decay(initializer=xavi_init,
            name="w_val", shape=[16, 1], wd=0.0008)
        advantage = tf.matmul(pre_adv, w_adv)
        value = tf.matmul(pre_val, w_val)

        # Combine them together to get final Q value
        q_out = value + tf.sub(advantage, tf.reduce_mean(advantage, axis=1,
                                                         keep_dims=True))

        return (q_out, tf.arg_max(q_out, 1), tf.reduce_max(q_out, 1))












