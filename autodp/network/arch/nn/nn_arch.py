"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from autodp.network.arch.base_arch import BaseArch
from autodp.utils.tf_utils import declare_variable
from autodp.utils.tf_utils import declare_variable_weight_decay
from autodp.config.cf_container import Config as cf
from autodp.utils.tf_utils import relu


@BaseArch.register
class NNArch(BaseArch):
    """
    A class inherited from the BaseArch to implement a complete CNN
    architecture.
    """
    def __init__(self, instance, name):
        """
        Initialize by storing the input instance.
        :param instance:
        :param name:
        """
        super().__init__(instance, name)

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
                name="weight", shape=[dim, cf.num_class], wd=cf.reg_coef)
            bias = declare_variable(name="bias", shape=[cf.num_class],
                                    initializer=self._conv_init)
            final_output = tf.nn.bias_add(tf.matmul(common_layer, weight), bias)

        return final_output































