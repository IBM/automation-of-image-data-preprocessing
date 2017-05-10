"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import tensorflow as tf

from sentana.graph.base_graph import BaseGraph
from sentana.config.cf_container import Config as cf


@BaseGraph.register
class SeqGraph(BaseGraph):
    """
    This class implements a sequential tensorflow graph.
    """
    def __init__(self, data_path):
        """
        Initialization of building a graph.
        :param data_path:
        """
        super().__init__(data_path)

    def _train(self, obj_func):
        """
        Define the train operator.
        :param obj_func: the function to be optimized
        :return: the train operator
        """
        # Create gradient steps
        tvars = tf.trainable_variables()
        grads = tf.gradients(obj_func, tvars)
        norm_grads, _ = tf.clip_by_global_norm(grads, cf.max_grad_norm)

        # Define a train step
        optimizer = tf.train.AdamOptimizer(cf.learning_rate)
        train_step = optimizer.apply_gradients(zip(norm_grads, tvars))

        return train_step

    def _build_model(self, data_path):
        """
        Build the total graph.
        :param data_path:
        :return:
        """
        self._instances, self._targets = self._declare_inputs(data_path)
        self._preds = self._inference(self._instances)
        self._obj_func = self._loss(self._preds, self._targets)
        self._train_step = self._train(self._obj_func)

