"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from autodp.network.arch.base_arch import BaseArch
from autodp.utils.tf_utils import declare_variable, declare_variable_weight_decay, relu
from autodp import cf


@BaseArch.register
class RLArch(BaseArch):
    """This class implements a normal network."""
    def __init__(self, instance, phase_train, keep_prob, name):
        """Initialize by storing the input instance."""
        super().__init__(instance, phase_train, keep_prob, name)

    def build_arch(self):
        """Build the concrete network for reinforcement learning."""
        # Build the common part
        common_layer = relu(self._build_common_part())
        dim = common_layer.get_shape().as_list()[1]

        # Add the specific part
        with tf.variable_scope("/".join([self._name, "final_layer"])) as scope:
            weight = declare_variable_weight_decay("weight", self._xavi_init, cf.reg_coef, [dim, cf.num_action])
            bias = declare_variable(name="bias", shape=[cf.num_action], initializer=self._conv_init)
            final_output = tf.nn.bias_add(tf.matmul(common_layer, weight), bias)
        return final_output, tf.arg_max(final_output, 1), tf.reduce_max(final_output, 1)
