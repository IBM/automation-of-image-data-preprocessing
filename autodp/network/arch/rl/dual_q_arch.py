import tensorflow as tf

from autodp.network.arch.base_arch import BaseArch
from autodp.utils.tf_utils import declare_variable_weight_decay
from autodp import cf


@BaseArch.register
class DualQArch(BaseArch):
    """This class implements the dual Q network."""
    def __init__(self, instance, phase_train, keep_prob, name):
        super().__init__(instance, phase_train, keep_prob, name)

    def build_arch(self):
        """Build the concrete dual Q network."""
        # Build the common part
        common_layer = self._build_common_part()
        dim = common_layer.get_shape().as_list()[1]

        # Add the specific part
        with tf.variable_scope("/".join([self._name, "final_layer"])) as scope:
            # Implement the dual Q network
            # Split into separate advantage and value
            pre_adv, pre_val = tf.split(value=common_layer, num_or_size_splits=2, axis=1)
            w_adv = declare_variable_weight_decay("w_adv", [dim/2, cf.num_action], self._xavi_init, cf.reg_coef)
            w_val = declare_variable_weight_decay("w_val", [dim/2, 1], self._xavi_init, cf.reg_coef)
            advantage = tf.matmul(pre_adv, w_adv)
            value = tf.matmul(pre_val, w_val)

            # Combine them together to get final Q value
            q_out = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keep_dims=True))
        return q_out, tf.arg_max(q_out, 1), tf.reduce_max(q_out, 1)
