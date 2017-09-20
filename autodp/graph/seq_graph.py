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
    def __init__(self):
        """
        Initialization of building a graph.
        """
        super().__init__()

    def _build_model(self):
        """
        Build the total graph.
        :param data_path:
        :param num_epoch:
        :return:
        """
        self._instances = tf.placeholder(tf.float32,
            shape=[None, cf.ima_height, cf.ima_width, 3])
        self._targets = tf.placeholder(tf.int32, shape=[None])
        self._preds = self._inference(self._instances)
        self._obj_func = self._loss(self._preds, self._targets)
        self._train_step = self._train(self._obj_func)

