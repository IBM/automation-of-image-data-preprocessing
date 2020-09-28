"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from autodp.runner.rl.rl_runner import RLRunner


def letsgo():
    """Main method to start testing RL training."""
    # Create the runner
    rl_runner = RLRunner()

    # Start training
    rl_runner.train_model(cont=False)

    # Start testing
    accuracy, reward, predict, actual = rl_runner.test_model()
    print("Final test accuracy is %g" % accuracy)


if __name__ == "__main__":
    letsgo()
