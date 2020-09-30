from autodp.runner.nn.nn_runner import NNRunner


if __name__ == "__main__":
    # Create the runner
    nn_runner = NNRunner()

    # Start training
    nn_runner.train_model()

    # Start testing
    accuracy = nn_runner.test_model()
    print("Final test accuracy is %g" % accuracy)
