"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from autodp.hptune.hptune import HPTune
from autodp.runner.rl.rl_runner import RLRunner


def letsgo():
    """
    Main method to start testing RL training.
    :return:
    """
    # Create the hyper-tuner
    hptune = HPTune()

    # Start tuning
    res = hptune.run_tuning(n_call=50)
    print("Best config: ", res.x)

    print("Start to train for the best configuration.........................")

    # Create the runner
    rl_runner = RLRunner()

    # Start training
    rl_runner.train_model(cont=False)

    # Start testing
    accuracy, reward, predict, actual = rl_runner.test_model()
    print("Final test accuracy is %g" % accuracy)


if __name__ == "__main__":
    letsgo()






























