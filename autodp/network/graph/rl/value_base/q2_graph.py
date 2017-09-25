"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from autodp.network.graph.base_graph import BaseGraph
from autodp.config.cf_container import Config as cf


@BaseGraph.register
class Q2Graph(BaseGraph):
    """
    This class implements a tensorflow graph for a Q-network with multiple
    output in reinforcement learning.
    """
    def __init__(self, net_arch, loss_func, tfreader=None):
        """
        Initialization of building a graph.
        :param net_arch:
        :param loss_func:
        :param tfreader:
        """
        super().__init__(net_arch, loss_func, tfreader)

    def _build_model(self, tfreader):
        """
        Build the total graph.
        :param tfreader: not used
        :return:
        """
        if tfreader is not None:
            raise ValueError("Building graph for a reinforcement learning task"
                             "is not supported with a tensorflow reader yet")
        else:
            # Declare network inputs
            self._instance = tf.placeholder(dtype=tf.float32,
                shape=[None, cf.ima_height, cf.ima_width, cf.ima_depth])
            self._action = tf.placeholder(tf.int32, shape=[None])
            self._target = tf.placeholder(tf.float32, shape=[None])

            # Build network architecture
            (self._qout, self._aout, self._qmax) = self._inference(
                self._instance)

            # Define objective function and train operator
            a_01 = tf.one_hot(self._actions, depth=cf.num_action,
                              axis=-1, dtype=tf.float32)
            self._pred = tf.reduce_sum(tf.multiply(self._qout, a_01), axis=1)
            self._train_loss(self._pred, self._target)

    @property
    def get_next_action(self):
        """
        Get best action for next step.
        :return:
        """
        return self._aout

    @property
    def get_current_action(self):
        """
        Get the current evaluated action.
        :return:
        """
        return self._action

    @property
    def get_qmax(self):
        """
        Get max Q value.
        :return:
        """
        return self._qmax

    @property
    def get_qout(self):
        """
        Get Q values.
        :return:
        """
        return self._qout

































