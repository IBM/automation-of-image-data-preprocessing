"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import ast
from autodp.config.cf_parser import config_parser


class Config(object):
    """
    This is a container for configured parameters.
    """
    # Load general parameters
    ima_height = int(config_parser.get("general_param", "ima_height"))
    ima_width = int(config_parser.get("general_param", "ima_width"))
    ima_depth = int(config_parser.get("general_param", "ima_depth"))
    num_class = int(config_parser.get("general_param", "num_class"))
    train_path = config_parser.get("general_param", "train_path")
    valid_path = config_parser.get("general_param", "valid_path")
    test_path = config_parser.get("general_param", "test_path")
    result_path = config_parser.get("general_param", "result_path")
    save_model = config_parser.get("general_param", "save_model")
    num_epoch = int(config_parser.get("general_param", "num_epoch"))
    batch_size = int(config_parser.get("general_param", "batch_size"))
    max_grad_norm = float(config_parser.get("general_param", "max_grad_norm"))

    learning_rate = float(config_parser.get("general_param", "learning_rate"))
    reg_coef = float(config_parser.get("general_param", "reg_coef"))
    kernel_size = ast.literal_eval(config_parser.get("general_param",
                                                     "kernel_size"))
    kernel_stride = ast.literal_eval(config_parser.get("general_param",
                                                     "kernel_stride"))
    pool_size = ast.literal_eval(config_parser.get("general_param",
                                                     "pool_size"))
    pool_stride = ast.literal_eval(config_parser.get("general_param",
                                                     "pool_stride"))
    fc_size = ast.literal_eval(config_parser.get("general_param",
                                                 "fc_size"))

    # Load neural network parameters
    valid_step = int(config_parser.get("nn_param", "valid_step"))
    nn_arch = config_parser.get("nn_param", "nn_arch")
    nn_loss = config_parser.get("nn_param", "nn_loss")
    nn_reader = config_parser.get("nn_param", "nn_reader")

    # Load reinforcement learning parameters
    rl_arch = config_parser.get("rl_param", "rl_arch")
    rl_loss = config_parser.get("rl_param", "rl_loss")
    rl_reader = config_parser.get("rl_param", "rl_reader")
    rl_graph = config_parser.get("rl_param", "rl_graph")
    rl_agent = config_parser.get("rl_param", "rl_agent")

    num_action = int(config_parser.get("rl_param", "num_action"))
    exploration = float(config_parser.get("rl_param", "exploration"))
    min_explore = float(config_parser.get("rl_param", "min_explore"))
    anneal_step = float(config_parser.get("rl_param", "anneal_step"))
    gamma = float(config_parser.get("rl_param", "gamma"))
    update_freq = int(config_parser.get("rl_param", "update_freq"))
    train_size = int(config_parser.get("rl_param", "train_size"))
    max_age = int(config_parser.get("rl_param", "max_age"))










