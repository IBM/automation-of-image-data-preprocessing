"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from sentana.app.vanila.model_runner import ModelRunner
from sentana.app.ril.reinforce_learning import ReInLearning


def letsgo():
    """
    Main method to start the ril version.
    :return:
    """
    # Create the runner
    ril = ReInLearning()

    # Start testing
    re_err, test_err = ril.test_and_analysis()
    print("Final valid reward and test error of ril version: %g and %g"
          % (-re_err, test_err))


if __name__ == "__main__":
    letsgo()


