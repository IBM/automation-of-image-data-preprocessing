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
        kern_init = tf.uniform_unit_scaling_initializer()
        bias_init = tf.uniform_unit_scaling_initializer(factor=0.000001)
        xavi_init = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("conv1") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[10, 10, 3, 20])
            conv = tf.nn.conv2d(tf.cast(self._instance, tf.float32),
                                kernel, [1, 1, 1, 1], "SAME")
            bias = declare_variable(name="bias", shape=[20],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool1 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv2") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[3, 3, 1, 10])
            conv = tf.nn.conv2d(pool1, kernel, [1, 2, 2, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[10],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool2 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv3") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[3, 3, 1, 10])
            conv = tf.nn.conv2d(pool2, kernel, [1, 2, 2, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[10],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool3 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv4") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[3, 3, 1, 5])
            conv = tf.nn.conv2d(pool3, kernel, [1, 2, 2, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[5],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool4 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv5") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[3, 3, 1, 5])
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[5],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool5 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv6") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[3, 3, 1, 5])
            conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[5],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool6 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv7") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[3, 3, 1, 5])
            conv = tf.nn.conv2d(pool6, kernel, [1, 1, 1, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[5],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool7 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv8") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[3, 3, 1, 5])
            conv = tf.nn.conv2d(pool7, kernel, [1, 1, 1, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[5],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool8 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv9") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[3, 3, 1, 5])
            conv = tf.nn.conv2d(pool8, kernel, [1, 1, 1, 1], padding="VALID")
            bias = declare_variable(name="bias", shape=[5],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool9 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv10") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[3, 3, 1, 5])
            conv = tf.nn.conv2d(pool9, kernel, [1, 1, 1, 1], padding="VALID")
            bias = declare_variable(name="bias", shape=[5],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool10 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("fc1") as scope:
            shape = pool10.get_shape().as_list()
            dim = shape[1] * shape[2]
            rsh = tf.reshape(pool10, [-1, dim])
            weights = declare_variable_weight_decay(initializer=xavi_init,
                name="weight", shape=[dim, 128], wd=0.0)
            bias = declare_variable(name="bias", shape=[128],
                                    initializer=bias_init)
            fc1 = relu(tf.nn.bias_add(tf.matmul(rsh, weights), bias))

        with tf.variable_scope("fc2") as scope:
            weights = declare_variable_weight_decay(initializer=xavi_init,
                name="weight", shape=[128, 64], wd=0.0)
            bias = declare_variable(name="bias", shape=[64],
                                    initializer=bias_init)
            fc2 = relu(tf.nn.bias_add(tf.matmul(fc1, weights), bias))

        with tf.variable_scope("fc3") as scope:
            weights = declare_variable_weight_decay(initializer=xavi_init,
                name="weight", shape=[64, 32], wd=0.0)
            bias = declare_variable(name="bias", shape=[32],
                                    initializer=bias_init)
            fc3 = relu(tf.nn.bias_add(tf.matmul(fc2, weights), bias))

        with tf.variable_scope("fc4") as scope:
            weights = declare_variable_weight_decay(initializer=xavi_init,
                name="weight", shape=[32, 24], wd=0.0)
            bias = declare_variable(name="bias", shape=[24],
                                    initializer=bias_init)
            fc4 = tf.nn.bias_add(tf.matmul(fc3, weights), bias)

        # Implement the Dual-Q network
        # Split into separate advantage and value
        pre_adv, pre_val = tf.split(value=fc4, num_split=2, split_dim=1)
        w_adv = declare_variable_weight_decay(initializer=xavi_init,
            name="w_adv", wd=0.0, shape=[12, cf.num_action])
        w_val = declare_variable_weight_decay(initializer=xavi_init,
            name="w_val", shape=[12, 1], wd=0.0)
        advantage = tf.matmul(pre_adv, w_adv)
        value = tf.matmul(pre_val, w_val)

        # Combine them together to get final Q value
        q_out = value + tf.sub(advantage, tf.reduce_mean(advantage, axis=1,
                                                         keep_dims=True))
        #q_out = tf.tanh(q_out)
        #q_out = tf.sigmoid(q_out)
        nom = q_out - tf.tile(tf.reshape(tf.reduce_min(q_out, 1),
                                         [-1, 1]), [1, 9])
        den = tf.tile(tf.reshape(tf.reduce_max(q_out, 1) - tf.reduce_min(
            q_out, 1), [-1, 1]), [1, 9])
        q_out = tf.div(nom, den)

        return (q_out, tf.arg_max(q_out, 1), tf.reduce_max(q_out, 1))












