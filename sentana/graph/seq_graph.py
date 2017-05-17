"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from sentana.graph.base_graph import BaseGraph
from sentana.reader.data_reader import DataReader


@BaseGraph.register
class SeqGraph(BaseGraph):
    """
    This class implements a sequential tensorflow graph.
    """
    def __init__(self, data_path=None):
        """
        Initialization of building a graph.
        :param data_path:
        """
        super().__init__(data_path)

    def _build_model(self, data_path):
        """
        Build the total graph.
        :param data_path:
        :return:
        """
        dr = DataReader(data_path)
        (self._instances, self._targets) = self._declare_inputs(dr)
        self._preds = self._inference(self._instances)
        self._obj_func = self._loss(self._preds, self._targets)
        self._train_step = self._train(self._obj_func)

