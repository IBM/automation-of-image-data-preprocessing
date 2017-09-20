"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from sentana.app.vanila.model_runner import ModelRunner
from sentana.app.ril.reinforce_learning import ReInLearning


def letsgo():
    """
    Main method to start the vanila version.
    :return:
    """
    # Create the runner
    mr = ModelRunner()

    # Start training
    mr.train_model(cont=False)

    # Start testing
    p, t = mr.test_model()
    print(p, t)

if __name__ == "__main__":
    letsgo()
