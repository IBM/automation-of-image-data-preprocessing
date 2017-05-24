"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from sentana.app.vanila.model_runner import ModelRunner
from sentana.app.ril.reinforce_learning import ReInLearning
from sentana.config.cf_container import Config as cf


def letstest():
    cf.num_epoch = 1
    #mr = ModelRunner()
    #pred_list, true_list = mr.test_model()
    #print((pred_list[:20], true_list[:20]))

    ril = ReInLearning()


if __name__ == "__main__":
    letstest()
