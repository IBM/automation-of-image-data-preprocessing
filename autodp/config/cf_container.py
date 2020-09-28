"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import ast

from autodp.config.cf_parser import config_parser, PATH_TO_CONFIG_FILE


class Config(object):
    """This is a container for configured parameters."""
    def __init__(self):
        # Load general parameters
        self.train_path = config_parser.get("general_param", "train_path")
        self.valid_path = config_parser.get("general_param", "valid_path")
        self.test_path = config_parser.get("general_param", "test_path")
        self.prep_path = config_parser.get("general_param", "prep_path")
        self.result_path = config_parser.get("general_param", "result_path")
        self.save_model = config_parser.get("general_param", "save_model")
        self.ima_height = int(config_parser.get("general_param", "ima_height"))
        self.ima_width = int(config_parser.get("general_param", "ima_width"))
        self.ima_depth = int(config_parser.get("general_param", "ima_depth"))
        self.num_class = int(config_parser.get("general_param", "num_class"))
        self.num_epoch = int(config_parser.get("general_param", "num_epoch"))
        self.batch_size = int(config_parser.get("general_param", "batch_size"))
        self.max_grad_norm = float(config_parser.get("general_param", "max_grad_norm"))
        self.valid_step = int(config_parser.get("general_param", "valid_step"))
        self.reader = config_parser.get("general_param", "reader")
        self.transfer = config_parser.getboolean("general_param", "transfer")
        self.keep_prob = float(config_parser.get("general_param", "keep_prob"))

        # Trainable via hypertuning
        self.learning_rate = float(config_parser.get("general_param", "learning_rate"))
        self.reg_coef = float(config_parser.get("general_param", "reg_coef"))
        self.kernel_size = ast.literal_eval(config_parser.get("general_param", "kernel_size"))
        self.kernel_stride = ast.literal_eval(config_parser.get("general_param", "kernel_stride"))
        self.pool_size = ast.literal_eval(config_parser.get("general_param", "pool_size"))
        self.pool_stride = ast.literal_eval(config_parser.get("general_param", "pool_stride"))
        self.fc_size = ast.literal_eval(config_parser.get("general_param", "fc_size"))

        # Load neural network parameters
        self.nn_arch = config_parser.get("nn_param", "nn_arch")
        self.nn_loss = config_parser.get("nn_param", "nn_loss")

        # Load reinforcement learning parameters
        self.rl_arch = config_parser.get("rl_param", "rl_arch")
        self.rl_loss = config_parser.get("rl_param", "rl_loss")
        self.rl_graph = config_parser.get("rl_param", "rl_graph")
        self.rl_agent = config_parser.get("rl_param", "rl_agent")
        self.rl_action = config_parser.get("rl_param", "rl_action")
        self.num_action = int(config_parser.get("rl_param", "num_action"))
        self.max_explore = float(config_parser.get("rl_param", "max_explore"))
        self.buf_size = int(config_parser.get("rl_param", "buf_size"))

        # Trainable via hypertuning
        self.min_explore = float(config_parser.get("rl_param", "min_explore"))
        self.anneal_step = float(config_parser.get("rl_param", "anneal_step"))
        self.gamma = float(config_parser.get("rl_param", "gamma"))
        self.max_age = int(config_parser.get("rl_param", "max_age"))
        self.dbl_coef = float(config_parser.get("rl_param", "dbl_coef"))

    def reset_config(self, path=PATH_TO_CONFIG_FILE):
        """Overwrite system config by application config."""
        config_parser.clear()
        config_parser.read(path)
        self.__init__()
