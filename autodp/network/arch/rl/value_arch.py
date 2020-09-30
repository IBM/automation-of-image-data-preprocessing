import tensorflow as tf

from autodp.network.arch.base_arch import BaseArch
from autodp.utils.tf_utils import declare_variable, declare_variable_weight_decay
from autodp import cf


@BaseArch.register
class ValueArch(BaseArch):
    """A class inherited from BaseArch to implement a complete architecture for the value network in policy gradient."""
    def __init__(self, instance, phase_train, keep_prob, name):
        super().__init__(instance, phase_train, keep_prob, name)

    def build_arch(self):
        """Build the concrete neural network architecture."""
        # Build the common part
        common_layer = tf.nn.relu(self._build_common_part())
        dim = common_layer.get_shape().as_list()[1]

        # Add the specific part
        with tf.variable_scope("/".join([self._name, "final_layer"])) as scope:
            weight = declare_variable_weight_decay("weight", self._xavi_init, cf.reg_coef, [dim, 1])
            bias = declare_variable(name="bias", shape=[1], initializer=self._conv_init)
            final_output = tf.nn.bias_add(tf.matmul(common_layer, weight), bias)
        return tf.reshape(final_output, [-1])
