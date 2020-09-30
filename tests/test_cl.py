import os

from autodp.runner.cl.cl_runner import CLRunner
from autodp.config.cf_container import Config as cf


if __name__ == "__main__":
    # Create the runner
    cl_runner = CLRunner(preprocess=True)

    # Start training
    cl_runner.train_model()

    # Start testing
    accuracy = cl_runner.test_model(path=os.path.join(cf.prep_path, "pp_test"))
    print("Final test accuracy is %g" % accuracy)
