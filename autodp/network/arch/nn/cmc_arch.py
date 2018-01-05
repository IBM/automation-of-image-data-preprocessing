"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf
import numpy as np

from autodp.network.arch.base_arch import BaseArch
from autodp.utils.tf_utils import declare_variable_weight_decay
from autodp.utils.tf_utils import declare_variable
from autodp.utils.tf_utils import batch_norm
from autodp import cf


@BaseArch.register
class CMCArch(BaseArch):
    """
    This class implements the Competitive Multi-scale Convolution network
    architecture proposed by Zhibin et al.
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
        # Define masking filters
        mask1 = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        mask3 = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        mask5 = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        mask7 = [[1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1]]

        # Build block 1
        # Layer 1
        bool_masks = [np.reshape([[mask1] * cf.ima_depth] * 192, [7, 7, cf.ima_depth, 192]),
                      np.reshape([[mask3] * cf.ima_depth] * 192, [7, 7, cf.ima_depth, 192]),
                      np.reshape([[mask5] * cf.ima_depth] * 192, [7, 7, cf.ima_depth, 192]),
                      np.reshape([[mask7] * cf.ima_depth] * 192, [7, 7, cf.ima_depth, 192])]
        layer1 = self._build_multi_conv_layer(layer_input=self._instance,
                                              kernel_size=[7, 7, cf.ima_depth, 192],
                                              kernel_stride=[1, 1, 1, 1],
                                              layer_name="multi_layer1",
                                              bool_masks=bool_masks)

        # Layer 2
        bool_masks = [np.reshape([[mask1] * 192] * 160, [7, 7, 192, 160]),
                      np.reshape([[mask3] * 192] * 160, [7, 7, 192, 160])]
        layer2 = self._build_multi_conv_layer(layer_input=layer1,
                                              kernel_size=[7, 7, 192, 160],
                                              kernel_stride=[1, 1, 1, 1],
                                              layer_name="multi_layer2",
                                              bool_masks=bool_masks)

        # Layer 3
        bool_masks = [np.reshape([[mask1] * 160] * 96, [7, 7, 160, 96]),
                      np.reshape([[mask3] * 160] * 96, [7, 7, 160, 96])]
        layer3 = self._build_multi_conv_layer(layer_input=layer2,
                                              kernel_size=[7, 7, 160, 96],
                                              kernel_stride=[1, 1, 1, 1],
                                              layer_name="multi_layer3",
                                              bool_masks=bool_masks)
        layer3_maxpool = tf.nn.max_pool(layer3, [1, 3, 3, 1],
                                        [1, 1, 1, 1], "VALID")
        layer3_dropout = tf.nn.dropout(layer3_maxpool, self._keep_prob)

        # Build block 2
        # Layer 4
        bool_masks = [np.reshape([[mask1] * 96] * 192, [7, 7, 96, 192]),
                      np.reshape([[mask3] * 96] * 192, [7, 7, 96, 192]),
                      np.reshape([[mask5] * 96] * 192, [7, 7, 96, 192]),
                      np.reshape([[mask7] * 96] * 192, [7, 7, 96, 192])]
        layer4 = self._build_multi_conv_layer(layer_input=layer3_dropout,
                                              kernel_size=[7, 7, 96, 192],
                                              kernel_stride=[1, 1, 1, 1],
                                              layer_name="multi_layer4",
                                              bool_masks=bool_masks)

        # Layer 5
        bool_masks = [np.reshape([[mask1] * 192] * 192, [7, 7, 192, 192]),
                      np.reshape([[mask3] * 192] * 192, [7, 7, 192, 192])]
        layer5 = self._build_multi_conv_layer(layer_input=layer4,
                                              kernel_size=[7, 7, 192, 192],
                                              kernel_stride=[1, 1, 1, 1],
                                              layer_name="multi_layer5",
                                              bool_masks=bool_masks)

        # Layer 6
        bool_masks = [np.reshape([[mask1] * 192] * 192, [7, 7, 192, 192]),
                      np.reshape([[mask3] * 192] * 192, [7, 7, 192, 192])]
        layer6 = self._build_multi_conv_layer(layer_input=layer5,
                                              kernel_size=[7, 7, 192, 192],
                                              kernel_stride=[1, 1, 1, 1],
                                              layer_name="multi_layer6",
                                              bool_masks=bool_masks)
        layer6_maxpool = tf.nn.max_pool(layer6, [1, 3, 3, 1],
                                        [1, 1, 1, 1], "VALID")
        layer6_dropout = tf.nn.dropout(layer6_maxpool, self._keep_prob)

        # Build block 3
        # Layer 7
        bool_masks = [np.reshape([[mask1] * 192] * 192, [7, 7, 192, 192]),
                      np.reshape([[mask3] * 192] * 192, [7, 7, 192, 192]),
                      np.reshape([[mask5] * 192] * 192, [7, 7, 192, 192])]
        layer7 = self._build_multi_conv_layer(layer_input=layer6_dropout,
                                              kernel_size=[7, 7, 192, 192],
                                              kernel_stride=[1, 1, 1, 1],
                                              layer_name="multi_layer7",
                                              bool_masks=bool_masks)

        # Layer 8
        bool_masks = [np.reshape([[mask1] * 192] * 192, [7, 7, 192, 192]),
                      np.reshape([[mask3] * 192] * 192, [7, 7, 192, 192])]
        layer8 = self._build_multi_conv_layer(layer_input=layer7,
                                              kernel_size=[7, 7, 192, 192],
                                              kernel_stride=[1, 1, 1, 1],
                                              layer_name="multi_layer8",
                                              bool_masks=bool_masks)

        # Layer 9
        bool_masks = [np.reshape([[mask1] * 192] * 10, [7, 7, 192, 10]),
                      np.reshape([[mask3] * 192] * 10, [7, 7, 192, 10])]
        layer9 = self._build_multi_conv_layer(layer_input=layer8,
                                              kernel_size=[7, 7, 192, 10],
                                              kernel_stride=[1, 1, 1, 1],
                                              layer_name="multi_layer9",
                                              bool_masks=bool_masks)
        layer9_avgpool = tf.nn.avg_pool(layer9,
            [1, cf.ima_height-4, cf.ima_width-4, 1], [1, 1, 1, 1], "VALID")

        return tf.squeeze(layer9_avgpool, [1, 2])

    def _build_multi_conv_layer(self, layer_input, kernel_size, kernel_stride,
                                bool_masks, layer_name):
        """
        Build a multi-convolutional layer.
        :param layer_input:
        :param kernel_size:
        :param kernel_stride:
        :param bool_masks:
        :param layer_name:
        :return:
        """
        norm = []
        with tf.variable_scope("/".join([self._name, layer_name])) as scope:
            for i in range(len(bool_masks)):
                kernel = tf.multiply(declare_variable_weight_decay(
                    shape=kernel_size, initializer=self._conv_init,
                    name="kernel_{}".format(i), wd=cf.reg_coef), bool_masks[i])
                conv = tf.nn.conv2d(layer_input, kernel, kernel_stride, "SAME")
                bias = declare_variable(name="bias_{}".format(i),
                                        shape=[kernel_size[-1]],
                                        initializer=self._conv_init)
                norm.append(tf.layers.batch_normalization(tf.nn.bias_add(conv, bias),
                                                          training=self._phase_train))

            layer_output = tf.reduce_max(tf.stack(norm), axis=0)

        return layer_output




























