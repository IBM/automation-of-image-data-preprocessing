"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from autodp.network.graph.base_graph import BaseGraph
from autodp import cf


@BaseGraph.register
class NNGraph(BaseGraph):
    """This class implements a vanila tensorflow graph for a plain neural network."""
    def __init__(self, net_arch, loss_func, name):
        """Initialization of building a graph."""
        super().__init__(net_arch, loss_func, name)

    def _build_model(self):
        """Build the total graph."""
        # The tensor inputs are defined
        if cf.transfer:
            self._instance = tf.placeholder(dtype=tf.float32, shape=[None, 1024])
        else:
            self._instance = tf.placeholder(dtype=tf.float32, shape=[None, cf.ima_height, cf.ima_width, cf.ima_depth])
        self._target = tf.placeholder(tf.int32, shape=[None])
        self._phase_train = tf.placeholder_with_default(False, shape=())
        self._keep_prob = tf.placeholder_with_default(1.0, shape=())

        # Define network architecture, objective function and train operator
        self._pred = self._inference(self._instance)
        self._train_loss(self._pred, self._target)

    def reset_train_step(self, var_list):
        """Reset train step with a new trainable variable list."""
        # Create gradient steps
        grads = tf.gradients(self._obj_func, var_list)
        norm_grads, _ = tf.clip_by_global_norm(grads, cf.max_grad_norm)

        # Define a train step
        optimizer = tf.train.AdamOptimizer(cf.learning_rate)
        self._train_step = optimizer.apply_gradients(zip(norm_grads, var_list))
