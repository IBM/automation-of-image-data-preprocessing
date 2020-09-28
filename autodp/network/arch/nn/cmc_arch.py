"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import numpy as np
import tensorflow as tf

from autodp.network.arch.base_arch import BaseArch
from autodp.utils.tf_utils import declare_variable_weight_decay, declare_variable
from autodp import cf


@BaseArch.register
class CMCArch(BaseArch):
    """This class implements the Competitive Multi-scale Convolution network architecture proposed by Zhibin et al."""
    def __init__(self, instance, phase_train, keep_prob, name):
        """Initialize by storing the input instance."""
        super().__init__(instance, phase_train, keep_prob, name)

    def build_arch(self):
        """Build the concrete neural network architecture."""
        # Define masking filters
        mask1 = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
        mask3 = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0],
                 [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
        mask5 = [[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0],
                 [0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]]
        mask7 = [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]

        # Build block 1
        b_m = [np.reshape([[mask1] * cf.ima_depth] * 192, [7, 7, cf.ima_depth, 192]),
               np.reshape([[mask3] * cf.ima_depth] * 192, [7, 7, cf.ima_depth, 192]),
               np.reshape([[mask5] * cf.ima_depth] * 192, [7, 7, cf.ima_depth, 192]),
               np.reshape([[mask7] * cf.ima_depth] * 192, [7, 7, cf.ima_depth, 192])]
        layer1 = self._build_multi_conv_layer(self._instance, [7, 7, cf.ima_depth, 192], [1, 1, 1, 1], "l1", b_m)

        b_m = [np.reshape([[mask1] * 192] * 160, [7, 7, 192, 160]), np.reshape([[mask3] * 192] * 160, [7, 7, 192, 160])]
        layer2 = self._build_multi_conv_layer(layer1, [7, 7, 192, 160], [1, 1, 1, 1], "l2", b_m)

        b_m = [np.reshape([[mask1] * 160] * 96, [7, 7, 160, 96]), np.reshape([[mask3] * 160] * 96, [7, 7, 160, 96])]
        layer3 = self._build_multi_conv_layer(layer2, [7, 7, 160, 96], [1, 1, 1, 1], "l3", b_m)
        layer3_maxpool = tf.nn.max_pool(layer3, [1, 3, 3, 1], [1, 1, 1, 1], "VALID")
        layer3_dropout = tf.nn.dropout(layer3_maxpool, self._keep_prob)

        # Build block 2
        b_m = [np.reshape([[mask1] * 96] * 192, [7, 7, 96, 192]), np.reshape([[mask3] * 96] * 192, [7, 7, 96, 192]),
               np.reshape([[mask5] * 96] * 192, [7, 7, 96, 192]), np.reshape([[mask7] * 96] * 192, [7, 7, 96, 192])]
        layer4 = self._build_multi_conv_layer(layer3_dropout, [7, 7, 96, 192], [1, 1, 1, 1], "l4", b_m)

        b_m = [np.reshape([[mask1] * 192] * 192, [7, 7, 192, 192]), np.reshape([[mask3] * 192] * 192, [7, 7, 192, 192])]
        layer5 = self._build_multi_conv_layer(layer4, [7, 7, 192, 192], [1, 1, 1, 1], "l5", b_m)

        b_m = [np.reshape([[mask1] * 192] * 192, [7, 7, 192, 192]), np.reshape([[mask3] * 192] * 192, [7, 7, 192, 192])]
        layer6 = self._build_multi_conv_layer(layer5, [7, 7, 192, 192], [1, 1, 1, 1], "l6", b_m)
        layer6_maxpool = tf.nn.max_pool(layer6, [1, 3, 3, 1], [1, 1, 1, 1], "VALID")
        layer6_dropout = tf.nn.dropout(layer6_maxpool, self._keep_prob)

        # Build block 3
        b_m = [np.reshape([[mask1] * 192] * 192, [7, 7, 192, 192]), np.reshape([[mask3] * 192] * 192, [7, 7, 192, 192]),
               np.reshape([[mask5] * 192] * 192, [7, 7, 192, 192])]
        layer7 = self._build_multi_conv_layer(layer6_dropout, [7, 7, 192, 192], [1, 1, 1, 1], "l7", b_m)

        b_m = [np.reshape([[mask1] * 192] * 192, [7, 7, 192, 192]), np.reshape([[mask3] * 192] * 192, [7, 7, 192, 192])]
        layer8 = self._build_multi_conv_layer(layer7, [7, 7, 192, 192], [1, 1, 1, 1], "l8", b_m)

        b_m = [np.reshape([[mask1] * 192] * 10, [7, 7, 192, 10]), np.reshape([[mask3] * 192] * 10, [7, 7, 192, 10])]
        layer9 = self._build_multi_conv_layer(layer8, [7, 7, 192, 10], [1, 1, 1, 1], "l9", b_m)
        layer9_avgpool = tf.nn.avg_pool(layer9, [1, cf.ima_height-4, cf.ima_width-4, 1], [1, 1, 1, 1], "VALID")
        return tf.squeeze(layer9_avgpool, [1, 2])

    def _build_multi_conv_layer(self, layer_input, kernel_size, kernel_stride, layer_name, bool_masks):
        """Build a multi-convolutional layer."""
        norm = []
        with tf.variable_scope("/".join([self._name, layer_name])) as scope:
            for i in range(len(bool_masks)):
                kernel = tf.multiply(declare_variable_weight_decay("kernel_{}".format(i), self._conv_init, cf.reg_coef,
                                                                   kernel_size,), bool_masks[i])
                conv = tf.nn.conv2d(layer_input, kernel, kernel_stride, "SAME")
                bias = declare_variable(name="bias_{}".format(i), shape=[kernel_size[-1]], initializer=self._conv_init)
                norm.append(tf.layers.batch_normalization(tf.nn.bias_add(conv, bias), training=self._phase_train))

            layer_output = tf.reduce_max(tf.stack(norm), axis=0)
        return layer_output
