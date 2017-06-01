"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from sentana.config.cf_parser import config_parser


class Config(object):
    """
    This is a container for configured parameters.
    """
    # Load general parameters
    ima_height = int(config_parser.get("general_param", "ima_height"))
    ima_width = int(config_parser.get("general_param", "ima_width"))
    num_class = int(config_parser.get("general_param", "num_class"))
    net_arch = config_parser.get("general_param", "net_arch")
    loss_func = config_parser.get("general_param", "loss_func")
    train_path = config_parser.get("general_param", "train_path")
    test_path = config_parser.get("general_param", "test_path")
    num_epoch = int(config_parser.get("general_param", "num_epoch"))
    batch_size = int(config_parser.get("general_param", "batch_size"))
    max_grad_norm = float(config_parser.get("general_param", "max_grad_norm"))
    learning_rate = float(config_parser.get("general_param", "learning_rate"))
    save_model = config_parser.get("general_param", "save_model")

    # Load ril parameters
    num_action = int(config_parser.get("ril_param", "num_action"))
    exploration = float(config_parser.get("ril_param", "exploration"))
    min_explore = float(config_parser.get("ril_param", "min_explore"))
    dec_explore = float(config_parser.get("ril_param", "dec_explore"))
    gamma = float(config_parser.get("ril_param", "gamma"))
    update_freq = int(config_parser.get("ril_param", "update_freq"))
    train_size = int(config_parser.get("ril_param", "train_size"))

