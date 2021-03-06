import tensorflow as tf

from autodp.network.graph.base_graph import BaseGraph
from autodp import cf


@BaseGraph.register
class PolicyGraph(BaseGraph):
    """This class implements a tensorflow graph for a policy-network for policy gradient method."""
    def __init__(self, net_arch, loss_func, name):
        super().__init__(net_arch, loss_func, name)

    def _build_model(self):
        # Declare network inputs
        self._instance = tf.placeholder(dtype=tf.float32, shape=[None, cf.ima_height, cf.ima_width, cf.ima_depth])
        self._action = tf.placeholder(tf.int32, shape=[None])
        self._target = tf.placeholder(tf.float32, shape=[None])
        self._phase_train = tf.placeholder_with_default(False, shape=())
        self._keep_prob = tf.placeholder_with_default(1.0, shape=())

        # Build network architecture
        (qout, _, _) = self._inference(self._instance)
        self._action_probs = tf.nn.softmax(qout)
        a_01 = tf.one_hot(self._action, depth=cf.num_action, axis=-1, dtype=tf.float32)
        self._action_prob = tf.reduce_sum(tf.multiply(self._action_probs, a_01), axis=1)

        # Loss and train op
        self._train_loss(self._action_prob, self._target)

    @property
    def get_action_probs(self):
        """Get probability of all actions."""
        return self._action_probs

    @property
    def get_action_prob(self):
        """Get probability of the current action."""
        return self._action_prob

    @property
    def get_current_action(self):
        """Get the current evaluated action."""
        return self._action
