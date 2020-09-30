import tensorflow as tf

from autodp.network.loss.loss import Loss


@Loss.register
class MSE(Loss):
    """This class implements the mean square error loss function."""
    def __init__(self, preds, trues):
        super().__init__(preds, trues)

    def _compute_loss(self, weights):
        """Compute the loss function."""
        return tf.contrib.losses.mean_squared_error(predictions=self._preds, labels=self._trues, weights=weights)
