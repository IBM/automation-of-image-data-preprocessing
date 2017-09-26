"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from autodp.network.graph.base_graph import BaseGraph
from autodp.config.cf_container import Config as cf


@BaseGraph.register
class NNGraph(BaseGraph):
    """
    This class implements a vanila tensorflow graph for a plain neural network.
    """
    def __init__(self, net_arch, loss_func, name, tfreader=None):
        """
        Initialization of building a graph.
        :param net_arch:
        :param loss_func:
        :param name:
        :param tfreader:
        """
        super().__init__(net_arch, loss_func, name, tfreader)

    def _build_model(self, tfreader):
        """
        Build the total graph.
        :param tfreader:
        :return:
        """
        # The tensor inputs are defined depending on the type of reader used.
        if tfreader is not None:
            (images, labels) = tfreader.get_batch(batch_size=cf.batch_size)
            self._instance = images
            self._target = labels
        else:
            self._instance = tf.placeholder(dtype=tf.float32,
                shape=[None, cf.ima_height, cf.ima_width, cf.ima_depth])
            self._target = tf.placeholder(tf.int32, shape=[None])

        # Define network architecture, objective function and train operator
        self._pred = self._inference(self._instance)
        self._train_loss(self._pred, self._target)
























