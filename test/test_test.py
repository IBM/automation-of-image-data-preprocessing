"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from sentana.app.vanila.model_runner import ModelRunner
from sentana.config.cf_container import Config as cf


def letstest():
    cf.num_epoch = 1
    mr = ModelRunner()
    pred_list, true_list = mr.test_model()

    print((pred_list[:10], true_list[:10]))


if __name__ == "__main__":
    letstest()
