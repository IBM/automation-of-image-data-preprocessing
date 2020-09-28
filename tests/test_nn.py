"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from autodp.runner.nn.nn_runner import NNRunner


def letsgo():
    """Main method to start testing CNN network."""
    # Create the runner
    nn_runner = NNRunner()

    # Start training
    nn_runner.train_model(cont=False)

    # Start testing
    accuracy = nn_runner.test_model()
    print("Final test accuracy is %g" % accuracy)


if __name__ == "__main__":
    letsgo()
