from autodp.hptune.hptune import HPTune
from autodp.runner.rl.rl_runner import RLRunner


if __name__ == "__main__":
    # Create the hyper-tuner
    hptune = HPTune()

    # Start tuning
    res = hptune.run_tuning(n_call=10)
    print("Best config: ", res.x)
    print("Start to train for the best configuration.........................")

    # Create the runner
    rl_runner = RLRunner()

    # Start training
    rl_runner.train_model()

    # Start testing
    accuracy, reward, predict, actual = rl_runner.test_model()
    print("Final test accuracy is %g" % accuracy)
