"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import json

from onebm_afl.config.cf_parser import config_parser


class Config(object):
    """
    This is a container for configured parameters.
    """
    # Load general parameters
    read_type = config_parser.get("general_param", "read_type")
    ima_height = int(config_parser.get("general_param", "ima_height"))
    ima_width = int(config_parser.get("general_param", "ima_width"))
    num_class = int(config_parser.get("general_param", "num_class"))
    net_arch = config_parser.get("general_param", "net_arch")
    loss_func = config_parser.get("general_param", "loss_func")


    data_path = config_parser.get("general_param", "data_path")
    test_path = config_parser.get("general_param", "test_path")
    num_tree = int(config_parser.get("general_param", "num_tree"))
    num_epoch = int(config_parser.get("general_param", "num_epoch"))
    batch_size = int(config_parser.get("general_param", "batch_size"))
    max_grad_norm = float(config_parser.get("general_param", "max_grad_norm"))
    learning_rate = float(config_parser.get("general_param", "learning_rate"))
    model_type = config_parser.get("general_param", "model_type")
    save_model = config_parser.get("general_param", "save_model")
    data_type = config_parser.get("general_param", "data_type")
    num_item = int(config_parser.get("general_param", "num_item"))
    num_response = int(config_parser.get("general_param", "num_response"))
    num_class = int(config_parser.get("general_param", "num_class"))
    max_num_feature = int(config_parser.get("general_param", "max_num_feature"))
    num_feature = json.loads(config_parser.get("general_param", "num_feature"))

    # Load embedding parameters
    embed_type = config_parser.get("embedding_param", "embed_type")
    embed_size = json.loads(config_parser.get("embedding_param", "embed_size"))
    schema = json.loads(config_parser.get("embedding_param", "schema"))

    # Load network parameters
    net_name = config_parser.get("network_param", "net_name")
    cell_type = config_parser.get("network_param", "cell_type")
    cell_size = int(config_parser.get("network_param", "cell_size"))

    # Load attention parameters
    att_type = config_parser.get("attention_param", "att_type")
    att_size = int(config_parser.get("attention_param", "att_size"))

    # Load prediction parameters
    act_func = config_parser.get("prediction_param", "act_func")
    layers = json.loads(config_parser.get("prediction_param", "layers"))
    keep_drop = json.loads(config_parser.get("prediction_param", "keep_drop"))
