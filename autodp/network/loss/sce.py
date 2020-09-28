"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from autodp.network.loss.loss import Loss
from autodp import cf


@Loss.register
class SCE(Loss):
    """This class implements the softmax cross entropy loss function."""
    def __init__(self, preds, trues):
        """Initialize by storing predicted and true values."""
        super().__init__(preds, trues)

    def _compute_loss(self, weights):
        """Compute the loss function."""
        onehot_labels = tf.one_hot(self._trues, depth=cf.num_class, axis=-1)
        return tf.losses.softmax_cross_entropy(logits=self._preds, onehot_labels=onehot_labels, weights=weights)
