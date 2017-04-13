"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf


class Loss(object):
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

    def _compute_loss(self, type, weights=1):
        """
        Compute the loss function.
        :param type:
        :return:
        """
        loss_funcs = ["sparse_softmax_cross_entropy", "log_loss", "hinge_loss",
                      "softmax_cross_entropy", "sigmoid_cross_entropy",
                      "cosine_distance", "mean_pairwise_squared_error",
                      "absolute_difference", "mean_squared_error"]

        if type not in loss_funcs:
            raise NotImplementedError("Loss function %s not supported." % type)

        module = __import__("".join(["tf.contrib.losses.", type]),
                            fromlist=type)
        loss_func = getattr(module, type)

        loss_func(predictions=self._preds, labels=self._trues, weights=weights)

    def compute_loss(self, type, weights=1):
        """
        Compute the final loss function.
        :param type:
        :return:
        """
        self._compute_loss(type=type, weights=weights)
        total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES),
                              name="total_loss")

        return total_loss

