"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from autodp.network.graph.base_graph import BaseGraph
from autodp.config.cf_container import Config as cf


@BaseGraph.register
class Q1Graph(BaseGraph):
    """
    This class implements a tensorflow graph for a Q-network with one value
    output in reinforcement learning.
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

    def _build_model(self, tfreader=None):
        """
        Build the total graph.
        :param tfreader: not used
        :return:
        """
        # Declare network inputs
        self._instance = tf.placeholder(dtype=tf.float32,
            shape=[None, cf.ima_height, cf.ima_width, cf.ima_depth])
        self._action = tf.placeholder(tf.int32, shape=[None])
        self._target = tf.placeholder(tf.float32, shape=[None])

        # Build network architecture
        self._pred = self._inference(self._instance)

        # Define objective function and train operator
        self._train_loss(self._pred, self._target)

    @property
    def get_current_action(self):
        """
        Get the current evaluated action.
        :return:
        """
        return self._action

































