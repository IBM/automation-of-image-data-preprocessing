"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import abc


class Loss(metaclass=abc.ABCMeta):
    """An abstract class to compute the loss function."""
    def __init__(self, preds, trues):
        """Initialize by storing predicted and true values."""
        self._preds = preds
        self._trues = trues

    @abc.abstractmethod
    def _compute_loss(self, weights):
        """Compute the loss function."""

    def compute_loss(self, weights=1):
        """Compute the final loss function."""
        return self._compute_loss(weights=weights)
