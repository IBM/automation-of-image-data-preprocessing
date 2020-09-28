"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from autodp.network.arch.base_arch import BaseArch
from autodp.utils.tf_utils import declare_variable, declare_variable_weight_decay, relu
from autodp import cf


@BaseArch.register
class KRArch(BaseArch):
    """A class inherited from the BaseArch to implement a complete CNN architecture."""
    def __init__(self, instance, phase_train, keep_prob, name):
        """Initialize by storing the input instance."""
        super().__init__(instance, phase_train, keep_prob, name)

    def build_arch(self):
        """Build the concrete neural network architecture."""
        kernel = declare_variable_weight_decay("kernel1", self._xavi_init, cf.reg_coef, cf.kernel_size[0])
        conv = tf.nn.conv2d(self._instance, kernel, [1, 1, 1, 1], "SAME")
        bias = declare_variable(shape=cf.kernel_size[0][-1], initializer=tf.constant_initializer(0.0), name="bias1")
        layer1 = relu(tf.nn.bias_add(conv, bias))

        kernel = declare_variable_weight_decay("kernel2", self._xavi_init, cf.reg_coef, cf.kernel_size[1],)
        conv = tf.nn.conv2d(layer1, kernel, [1, 1, 1, 1], "VALID")
        bias = declare_variable(shape=cf.kernel_size[1][-1], initializer=tf.constant_initializer(0.0), name="bias2")
        layer2 = tf.nn.dropout(tf.nn.max_pool(relu(tf.nn.bias_add(conv, bias)), [1, 2, 2, 1], [1, 2, 2, 1], "VALID"),
                               self._keep_prob)

        kernel = declare_variable_weight_decay("kernel3", self._xavi_init, cf.reg_coef, cf.kernel_size[2])
        conv = tf.nn.conv2d(layer2, kernel, [1, 1, 1, 1], "SAME")
        bias = declare_variable(shape=cf.kernel_size[2][-1], initializer=tf.constant_initializer(0.0), name="bias3")
        layer3 = relu(tf.nn.bias_add(conv, bias))

        kernel = declare_variable_weight_decay("kernel4", self._xavi_init, cf.reg_coef, cf.kernel_size[3])
        conv = tf.nn.conv2d(layer3, kernel, [1, 1, 1, 1], "VALID")
        bias = declare_variable(shape=cf.kernel_size[3][-1], initializer=tf.constant_initializer(0.0), name="bias4")
        layer4 = tf.nn.dropout(tf.nn.max_pool(relu(tf.nn.bias_add(conv, bias)), [1, 2, 2, 1], [1, 2, 2, 1], "VALID"),
                               self._keep_prob)

        kernel = declare_variable_weight_decay("kernel5", self._xavi_init, cf.reg_coef, cf.kernel_size[4])
        conv = tf.nn.conv2d(layer4, kernel, [1, 1, 1, 1], "SAME")
        bias = declare_variable(shape=cf.kernel_size[4][-1], initializer=tf.constant_initializer(0.0), name="bias5")
        layer5 = relu(tf.nn.bias_add(conv, bias))

        kernel = declare_variable_weight_decay("kernel6", self._xavi_init, cf.reg_coef, cf.kernel_size[5])
        conv = tf.nn.conv2d(layer5, kernel, [1, 1, 1, 1], "VALID")
        bias = declare_variable(shape=cf.kernel_size[5][-1], initializer=tf.constant_initializer(0.0), name="bias6")
        layer6 = tf.nn.dropout(tf.nn.max_pool(relu(tf.nn.bias_add(conv, bias)), [1, 2, 2, 1], [1, 2, 2, 1], "VALID"),
                               self._keep_prob)

        shape = layer6.get_shape().as_list()
        dim = shape[1] * shape[2] * shape[3]
        l_input = tf.reshape(layer6, [-1, dim])

        weight = declare_variable_weight_decay("kernel7", self._xavi_init, cf.reg_coef, [dim, cf.fc_size[0]])
        bias = declare_variable(shape=[cf.fc_size[0]], initializer=tf.constant_initializer(0.0), name="bias7")
        layer7 = tf.nn.dropout(relu(tf.nn.bias_add(tf.matmul(l_input, weight), bias)), self._keep_prob)

        weight = declare_variable_weight_decay("kernel8", self._xavi_init, cf.reg_coef, [cf.fc_size[0], cf.fc_size[1]])
        bias = declare_variable(shape=[cf.fc_size[1]], initializer=tf.constant_initializer(0.0), name="bias8")
        layer8 = tf.nn.dropout(relu(tf.nn.bias_add(tf.matmul(layer7, weight), bias)), self._keep_prob)

        weight = declare_variable_weight_decay("kernel9", self._xavi_init, cf.reg_coef, [cf.fc_size[1], cf.num_class])
        bias = declare_variable(shape=[cf.num_class], initializer=tf.constant_initializer(0.0), name="bias9")
        final_output = tf.nn.bias_add(tf.matmul(layer8, weight), bias)

        return final_output
