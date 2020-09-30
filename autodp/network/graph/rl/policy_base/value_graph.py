import tensorflow as tf

from autodp.network.graph.base_graph import BaseGraph
from autodp import cf


@BaseGraph.register
class ValueGraph(BaseGraph):
    """This class implements a tensorflow graph for a value-network for policy gradient method."""
    def __init__(self, net_arch, loss_func, name):
        super().__init__(net_arch, loss_func, name)

    def _build_model(self):
        # Declare network inputs
        self._instance = tf.placeholder(dtype=tf.float32, shape=[None, cf.ima_height, cf.ima_width, cf.ima_depth])
        self._target = tf.placeholder(tf.float32, shape=[None])
        self._phase_train = tf.placeholder_with_default(False, shape=())
        self._keep_prob = tf.placeholder_with_default(1.0, shape=())

        # Build network architecture
        self._value = self._inference(self._instance)

        # Loss and train op
        self._train_loss(self._value, self._target)

    @property
    def get_value(self):
        """Get value of a state."""
        return self._value
