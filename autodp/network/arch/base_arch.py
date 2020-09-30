import abc
import tensorflow as tf

from autodp.utils.tf_utils import declare_variable, declare_variable_weight_decay, relu
from autodp import cf


class BaseArch(metaclass=abc.ABCMeta):
    """A base class to build a concrete architecture."""
    def __init__(self, instance, phase_train, keep_prob, name):
        self._instance = instance
        self._name = name
        self._phase_train = phase_train
        self._keep_prob = keep_prob
        self._conv_init = tf.truncated_normal_initializer(stddev=0.1)
        self._xavi_init = tf.contrib.layers.xavier_initializer()

    def _build_conv_layer(self, layer_input, kernel_size, kernel_stride, pool_size, pool_stride, layer_name):
        with tf.variable_scope("/".join([self._name, layer_name])) as scope:
            kernel = declare_variable_weight_decay("kernel", self._conv_init, cf.reg_coef, kernel_size)
            conv = tf.nn.conv2d(layer_input, kernel, kernel_stride, "SAME")
            bias = declare_variable(name="bias", shape=[kernel_size[-1]], initializer=self._conv_init)
            norm = tf.nn.lrn(input=relu(tf.nn.bias_add(conv, bias)))
            layer_output = tf.nn.max_pool(norm, pool_size, pool_stride, "VALID")
        return layer_output

    def _build_fully_connected_layer(self, layer_input, fc_size, layer_name):
        with tf.variable_scope("/".join([self._name, layer_name])) as scope:
            dim = layer_input.get_shape().as_list()[1]
            weight = declare_variable_weight_decay("weight", self._xavi_init, cf.reg_coef, [dim, fc_size])
            bias = declare_variable(name="bias", shape=[fc_size], initializer=self._conv_init)
            layer_output = tf.nn.bias_add(tf.matmul(relu(layer_input), weight), bias)
        return layer_output

    def _build_common_part(self):
        """Build the common part of network architectures used by nn and rl packages."""
        l_input = tf.cast(self._instance, tf.float32)

        # Build convolutional layers
        if len(self._instance.get_shape().as_list()) > 2:
            for i in range(len(cf.kernel_size)):
                l_input = self._build_conv_layer(l_input, cf.kernel_size[i], cf.kernel_stride[i], cf.pool_size[i],
                                                 cf.pool_stride[i], "conv{}".format(i))
            shape = l_input.get_shape().as_list()
            dim = shape[1] * shape[2] * shape[3]
            l_input = tf.reshape(l_input, [-1, dim])

        # Build fully connected layers
        for i in range(len(cf.fc_size)):
            l_input = self._build_fully_connected_layer(l_input, cf.fc_size[i], "fc{}".format(i))
        return l_input

    @abc.abstractmethod
    def build_arch(self):
        """Build a concrete architecture."""
