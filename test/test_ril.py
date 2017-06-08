"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from sentana.app.vanila.model_runner import ModelRunner
from sentana.app.ril.reinforce_learning import ReInLearning


def letsgo():
    ril = ReInLearning()
    ril.train_policy(cont=False)
    test_err, _, _ = ril.test_policy()
    print("Final test error of ril version after training: %g" % test_err)


if __name__ == "__main__":
    letsgo()
