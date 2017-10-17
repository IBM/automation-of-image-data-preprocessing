"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os

from autodp.runner.cl.cl_runner import CLRunner
from autodp.config.cf_container import Config as cf


def letsgo():
    """
    Main method to start testing RL training.
    :return:
    """
    # Create the runner
    cl_runner = CLRunner(preprocess=True)

    # Start training
    cl_runner.train_model(cont=False)

    # Start testing
    accuracy = cl_runner.test_model(path=os.path.join(cf.prep_path, "pp_test"))
    print("Final test accuracy is %g" % accuracy)


if __name__ == "__main__":
    letsgo()























