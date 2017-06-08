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
    def __init__(self, data_path=None, num_epoch=1):
        """
        Initialization of building a graph.
        :param data_path:
        :param num_epoch:
        """
        super().__init__(data_path, num_epoch)

    def _build_model(self, data_path, num_epoch):
        """
        Build the total graph.
        :param data_path:
        :param num_epoch:
        :return:
        """
        dr = DataReader(data_path, num_epoch)
        (self._instances, self._targets) = self._declare_inputs(dr)
        self._preds = self._inference(self._instances)
        self._obj_func = self._loss(self._preds, self._targets)
        self._train_step = self._train(self._obj_func)

