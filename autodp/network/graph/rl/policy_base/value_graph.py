"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from autodp.network.graph.base_graph import BaseGraph
from autodp import cf


@BaseGraph.register
class ValueGraph(BaseGraph):
    """
    This class implements a tensorflow graph for a value-network for policy
    gradient method in reinforcement learning.
    """
    def __init__(self, net_arch, loss_func, name):
        """
        Initialization of building a graph.
        :param net_arch:
        :param loss_func:
        :param name:
        """
        super().__init__(net_arch, loss_func, name)

    def _build_model(self):
        """
        Build the total graph.
        :return:
        """
        # Declare network inputs
        self._instance = tf.placeholder(dtype=tf.float32,
                                        shape=[None, cf.ima_height,
                                               cf.ima_width, cf.ima_depth])
        self._action = tf.placeholder(tf.int32, shape=[None])
        self._target = tf.placeholder(tf.float32, shape=[None])

        # Build network architecture
        (qout, _, _) = self._inference(self._instance)
        self._action_probs = tf.squeeze(tf.nn.softmax(qout))
        self._action_prob = tf.gather(self._action_probs, self._action)

        # Loss and train op
        self._train_loss(self._action_prob, self._target)

    @property
    def get_action_probs(self):
        """
        Get probability of all actions.
        :return:
        """
        return self._action_probs

    @property
    def get_action_prob(self):
        """
        Get probability of the current action.
        :return:
        """
        return self._action_prob
