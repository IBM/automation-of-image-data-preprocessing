"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from autodp.network.arch.base_arch import BaseArch
from autodp.utils.tf_utils import declare_variable
from autodp.utils.tf_utils import declare_variable_weight_decay
from autodp import cf
from autodp.utils.tf_utils import relu


@BaseArch.register
class ValueArch(BaseArch):
    """
    A class inherited from the BaseArch to implement a complete architecture
    for the value network in policy gradient.
    """
    def __init__(self, instance, phase_train, keep_prob, name):
        """
        Initialize by storing the input instance.
        :param instance:
        :param phase_train:
        :param keep_prob:
        :param name:
        """
        super().__init__(instance, phase_train, keep_prob, name)

    def build_arch(self):
        """
        Build the concrete neural network architecture.
        :return:
        """
        # Build the common part
        common_layer = relu(self._build_common_part())
        dim = common_layer.get_shape().as_list()[1]

        # Add the specific part
        with tf.variable_scope("/".join([self._name, "final_layer"])) as scope:
            weight = declare_variable_weight_decay(initializer=self._xavi_init,
                name="weight", shape=[dim, 1], wd=cf.reg_coef)
            bias = declare_variable(name="bias", shape=[1],
                                    initializer=self._conv_init)
            final_output = tf.nn.bias_add(tf.matmul(common_layer, weight), bias)

        return tf.reshape(final_output, [-1])







































