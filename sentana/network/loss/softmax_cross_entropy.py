"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from sentana.network.loss.loss import Loss


@Loss.register
class SoftmaxCrossEntropy(Loss):
    """
    This class implements the softmax_cross_entropy loss function.
    """
    def __init__(self, preds, trues):
        """
        Initialize by storing predicted and true values.
        :param preds:
        :param trues:
        """
        super().__init__(preds, trues)

    def _compute_loss(self, weights=1):
        """
        Compute the loss function.
        :param type:
        :return:
        """
        onehot_labels = tf.one_hot(self._trues, depth=2, axis=1)
        tf.contrib.losses.softmax_cross_entropy(logits=self._preds,
                                                onehot_labels=onehot_labels,
                                                weights=weights)
