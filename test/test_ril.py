"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from sentana.app.vanila.model_runner import ModelRunner
from sentana.app.ril.reinforce_learning import ReInLearning


def letsgo():
    ril = ReInLearning()
    ril.train_policy(cont=False)
    re_err, test_err, _, _ = ril.test_policy()
    print("Final valid reward and test error of ril version: %g"
          % (-re_err, test_err))


if __name__ == "__main__":
    letsgo()
