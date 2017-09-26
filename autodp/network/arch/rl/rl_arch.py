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
class DualQArch(BaseArch):
    """
    This class implements the dual Q network.
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
        Build the concrete dual Q network.
        :return:
        """
        # Build the common part
        common_layer = self._build_common_part()
        dim = common_layer.get_shape().as_list()[1]

        # Add the specific part
        # Implement the dual Q network
        # Split into separate advantage and value
        pre_adv, pre_val = tf.split(value=common_layer, num_split=2,
                                    split_dim=1)
        w_adv = declare_variable_weight_decay(initializer=self._xavi_init,
            name="w_adv", wd=cf.reg_coef, shape=[dim/2, cf.num_action])
        w_val = declare_variable_weight_decay(initializer=self._xavi_init,
            name="w_val", shape=[dim/2, 1], wd=cf.reg_coef)
        advantage = tf.matmul(pre_adv, w_adv)
        value = tf.matmul(pre_val, w_val)

        # Combine them together to get final Q value
        q_out = value + tf.sub(advantage, tf.reduce_mean(advantage, axis=1,
                                                         keep_dims=True))

        return (q_out, tf.arg_max(q_out, 1), tf.reduce_max(q_out, 1))


































