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
        bias_init = tf.uniform_unit_scaling_initializer(factor=0.0000001)
        xavi_init = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("conv1") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[11, 11, 3, 96])
            conv = tf.nn.conv2d(tf.cast(self._instance, tf.float32),
                                kernel, [1, 2, 2, 1], "VALID")
            bias = declare_variable(name="bias", shape=[96],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool1 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv2") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 256])
            conv = tf.nn.conv2d(pool1, kernel, [1, 2, 2, 1], padding="VALID")
            bias = declare_variable(name="bias", shape=[256],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool2 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv3") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool3 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv4") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool4 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv5") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool5 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv6") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool6 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv7") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool6, kernel, [1, 1, 1, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool7 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv8") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool7, kernel, [1, 1, 1, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool8 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv9") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool8, kernel, [1, 1, 1, 1], padding="VALID")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool9 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv10") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool9, kernel, [1, 1, 1, 1], padding="VALID")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool10 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv11") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool10, kernel, [1, 1, 1, 1], padding="VALID")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool11 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv12") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool11, kernel, [1, 1, 1, 1], padding="VALID")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool12 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv13") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool12, kernel, [1, 1, 1, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool13 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv14") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool13, kernel, [1, 1, 1, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool14 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv15") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool14, kernel, [1, 1, 1, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool15 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv16") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool15, kernel, [1, 1, 1, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool16 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv17") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool16, kernel, [1, 1, 1, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool17 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv18") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool17, kernel, [1, 1, 1, 1], padding="SAME")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool18 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv19") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool18, kernel, [1, 1, 1, 1], padding="VALID")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool19 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("conv20") as scope:
            kernel = declare_variable_weight_decay(initializer=kern_init,
                name="kernel", wd=0.0, shape=[5, 5, 1, 50])
            conv = tf.nn.conv2d(pool19, kernel, [1, 1, 1, 1], padding="VALID")
            bias = declare_variable(name="bias", shape=[50],
                                    initializer=bias_init)
            pool = tf.reduce_max(relu(tf.nn.bias_add(conv, bias)),
                reduction_indices=[3], keep_dims=True, name="pool")
            pool20 = tf.nn.lrn(input=pool, name="norm_pool")

        with tf.variable_scope("fc1") as scope:
            shape = pool20.get_shape().as_list()
            dim = shape[1] * shape[2]
            rsh = tf.reshape(pool20, [-1, dim])
            weights = declare_variable_weight_decay(initializer=xavi_init,
                name="weight", shape=[dim, 512], wd=0.0)
            bias = declare_variable(name="bias", shape=[512],
                                    initializer=bias_init)
            fc1 = relu(tf.nn.bias_add(tf.matmul(rsh, weights), bias))

        with tf.variable_scope("fc2") as scope:
            weights = declare_variable_weight_decay(initializer=xavi_init,
                name="weight", shape=[512, 512], wd=0.0)
            bias = declare_variable(name="bias", shape=[512],
                                    initializer=bias_init)
            fc2 = relu(tf.nn.bias_add(tf.matmul(fc1, weights), bias))

        with tf.variable_scope("fc3") as scope:
            weights = declare_variable_weight_decay(initializer=xavi_init,
                name="weight", shape=[512, 32], wd=0.0)
            bias = declare_variable(name="bias", shape=[32],
                                    initializer=bias_init)
            fc3 = relu(tf.nn.bias_add(tf.matmul(fc2, weights), bias))

        with tf.variable_scope("fc4") as scope:
            weights = declare_variable_weight_decay(initializer=xavi_init,
                name="weight", shape=[32, 2], wd=0.0)
            bias = declare_variable(name="bias", shape=[2],
                                    initializer=bias_init)
            fc4 = tf.nn.bias_add(tf.matmul(fc3, weights), bias)

        return fc4









