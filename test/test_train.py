"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from sentana.runner.model_runner import ModelRunner


def letstrain():
    mr = ModelRunner()
    mr.train_model(cont=True)


if __name__ == "__main__":
    letstrain()
