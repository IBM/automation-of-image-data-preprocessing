"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import abc

import tensorflow as tf


class Loss(metaclass=abc.ABCMeta):
    """
    A class to compute the loss function.
    """
    def __init__(self, preds, trues):
        """
        Initialize by storing predicted and true values.
        :param preds:
        :param trues:
        """
        self._preds = preds
        self._trues = trues

    @abc.abstractmethod
    def _compute_loss(self, weights=1):
        """
        Compute the loss function.
        :param type:
        :return:
        """
        # loss_funcs = ["sparse_softmax_cross_entropy", "log_loss", "hinge_loss",
        #               "softmax_cross_entropy", "sigmoid_cross_entropy",
        #               "cosine_distance", "mean_pairwise_squared_error",
        #               "absolute_difference", "mean_squared_error"]
        #
        # if type not in loss_funcs:
        #     raise NotImplementedError("Loss function %s not supported." % type)
        #
        # module = __import__("tensorflow.contrib.losses", fromlist=type)
        # loss_func = getattr(module, type)
        #
        # onehot_labels = tf.one_hot(self._trues, depth=2, axis=1)
        # loss_func(logits=self._preds, onehot_labels=onehot_labels,
        #           weights=weights)

    def compute_loss(self, weights=1):
        """
        Compute the final loss function.
        :param type:
        :return:
        """
        total_loss = self._compute_loss(weights=weights)
        #total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES),
        #                      name="total_loss")

        return total_loss
