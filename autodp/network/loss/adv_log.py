import tensorflow as tf

from autodp.network.loss.loss import Loss


@Loss.register
class AdvLog(Loss):
    """This class implements the advantage log loss function used in policy gradient strategy."""
    def __init__(self, preds, trues):
        super().__init__(preds, trues)

    def _compute_loss(self, weights):
        """Compute the loss function."""
        return tf.reduce_sum(-tf.log(self._preds) * self._trues)
