"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from autodp.network.loss.loss import Loss


@Loss.register
class AdvLog(Loss):
    """This class implements the advantage log loss function used in policy gradient strategy."""
    def __init__(self, preds, trues):
        """Initialize by storing predicted and true values."""
        super().__init__(preds, trues)

    def _compute_loss(self, weights):
        """Compute the loss function."""
        loss = tf.reduce_sum(-tf.log(self._preds) * self._trues)
        return loss
