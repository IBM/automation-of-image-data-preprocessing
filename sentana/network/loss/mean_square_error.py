"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from sentana.network.loss.loss import Loss


@Loss.register
class MeanSquareError(Loss):
    """
    This class implements the mean_square_error loss function.
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
        tf.contrib.losses.mean_squared_error(predictions=self._preds,
                                             targets=self._trues,
                                             weights=weights)

