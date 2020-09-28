"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from autodp.network.graph.base_graph import BaseGraph
from autodp import cf


@BaseGraph.register
class Q2Graph(BaseGraph):
    """This class implements a tensorflow graph for a Q-network with multiple output in reinforcement learning."""
    def __init__(self, net_arch, loss_func, name):
        """Initialization of building a graph."""
        super().__init__(net_arch, loss_func, name)

    def _build_model(self):
        """Build the total graph."""
        # Declare network inputs
        self._instance = tf.placeholder(dtype=tf.float32, shape=[None, 1024])
        self._action = tf.placeholder(tf.int32, shape=[None])
        self._target = tf.placeholder(tf.float32, shape=[None])
        self._phase_train = tf.placeholder_with_default(False, shape=())
        self._keep_prob = tf.placeholder_with_default(1.0, shape=())

        # Build network architecture
        (self._qout, self._aout, self._qmax) = self._inference(self._instance)

        # Define objective function and train operator
        a_01 = tf.one_hot(self._action, depth=cf.num_action, axis=-1, dtype=tf.float32)
        self._pred = tf.reduce_sum(tf.multiply(self._qout, a_01), axis=1)
        self._train_loss(self._pred, self._target)

    @property
    def get_next_action(self):
        """Get best action for next step."""
        return self._aout

    @property
    def get_current_action(self):
        """Get the current evaluated action."""
        return self._action

    @property
    def get_qmax(self):
        """Get max Q value."""
        return self._qmax

    @property
    def get_qout(self):
        """Get Q values."""
        return self._qout
