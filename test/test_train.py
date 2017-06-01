"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from sentana.app.vanila.model_runner import ModelRunner
from sentana.app.ril.reinforce_learning import ReInLearning


def letstrain():
    #mr = ModelRunner()
    #mr.train_model(cont=False)

    ril = ReInLearning()
    ril.train_policy(cont=False)


if __name__ == "__main__":
    letstrain()
