"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import abc
import tensorflow as tf

from autodp.utils.tf_utils import declare_variable_weight_decay
from autodp.utils.tf_utils import declare_variable
from autodp.config.cf_container import Config as cf
from autodp.utils.tf_utils import relu


class BaseArch(metaclass=abc.ABCMeta):
    """
    A base class to build a concrete architecture.
    """
    def __init__(self, instance, name):
        """
        Initialize by storing the input instance.
        :param instance:
        """
        self._instance = instance
        self._name = name
        self._conv_init = tf.truncated_normal_initializer(stddev=0.1)
        self._xavi_init = tf.contrib.layers.xavier_initializer()

    def _build_conv_layer(self, layer_input, kernel_size, kernel_stride,
                          pool_size, pool_stride, layer_name):
        """
        Build a convolutional layer.
        :param layer_input:
        :param kernel_size:
        :param kernel_stride:
        :param pool_size:
        :param pool_stride:
        :param layer_name:
        :return:
        """
        with tf.variable_scope("/".join([self._name, layer_name])) as scope:
            kernel = declare_variable_weight_decay(initializer=self._conv_init,
                name="kernel", wd=cf.reg_coef, shape=kernel_size)
            conv = tf.nn.conv2d(layer_input, kernel, kernel_stride, "SAME")
            bias = declare_variable(name="bias", shape=[kernel_size[-1]],
                                    initializer=self._conv_init)
            norm = tf.nn.lrn(input=relu(tf.nn.bias_add(conv, bias)))
            layer_output = tf.nn.max_pool(norm, pool_size, pool_stride, "VALID")

        return layer_output

    def _build_fully_connected_layer(self, layer_input, fc_size, layer_name):
        """
        Build a fully connected layer.
        :param layer_input:
        :param fc_size:
        :param layer_name:
        :return:
        """
        with tf.variable_scope("/".join([self._name, layer_name])) as scope:
            dim = layer_input.get_shape().as_list()[1]
            weight = declare_variable_weight_decay(initializer=self._xavi_init,
                name="weight", shape=[dim, fc_size], wd=cf.reg_coef)
            bias = declare_variable(name="bias", shape=[fc_size],
                                    initializer=self._conv_init)
            layer_output = tf.nn.bias_add(tf.matmul(relu(layer_input),
                                                    weight), bias)

        return layer_output

    def _build_common_part(self):
        """
        Build the common part of network architectures used by nn and rl
        packages.
        :return:
        """
        # Build convolutional layers
        l_input = tf.cast(self._instance, tf.float32)
        for i in range(len(cf.kernel_size)):
            l_input = self._build_conv_layer(layer_input=l_input,
                                             kernel_size=cf.kernel_size[i],
                                             kernel_stride=cf.kernel_stride[i],
                                             pool_size=cf.pool_size[i],
                                             pool_stride=cf.pool_stride[i],
                                             layer_name="conv{}".format(i))

        # Build fully connected layers
        shape = l_input.get_shape().as_list()
        dim = shape[1] * shape[2] * shape[3]
        l_input = tf.reshape(l_input, [-1, dim])
        for i in range(len(cf.fc_size)):
            l_input = self._build_fully_connected_layer(layer_input=l_input,
                fc_size=cf.fc_size[i], layer_name="fc{}".format(i))

        return l_input

    @abc.abstractmethod
    def build_arch(self):
        """
        Build a concrete architecture.
        :return:
        """









































