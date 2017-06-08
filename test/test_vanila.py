"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from sentana.app.vanila.model_runner import ModelRunner
from sentana.app.ril.reinforce_learning import ReInLearning


def letsgo():
    mr = ModelRunner()
    mr.train_model(cont=False)
    test_err, _, _ = mr.test_model()
    print("Final test error of vanila version after training: %g" % test_err)


if __name__ == "__main__":
    letsgo()
