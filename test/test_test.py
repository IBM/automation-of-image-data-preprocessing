"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from sentana.app.vanila.model_runner import ModelRunner
from sentana.app.ril.reinforce_learning import ReInLearning
from sentana.config.cf_container import Config as cf


def letstest():
    #cf.num_epoch = 1
    #mr = ModelRunner()
    #pred_list, true_list = mr.test_model()
    #print((pred_list[:], true_list[:]))

    ril = ReInLearning()
    pred_list, true_list = ril.test_policy()
    print((pred_list[:50], true_list[:50]))
    print(len(pred_list), len(true_list))


if __name__ == "__main__":
    letstest()
