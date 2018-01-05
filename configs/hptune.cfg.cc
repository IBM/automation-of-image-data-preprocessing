[general_param]
method=gp

[objective]
module=autodp.runner.rl.rl_runner
function=RLRunner

[config]
output=/u/minhtn/projects/autodp/configs/autodp.cfg
ima_depth=1
search_space={
			"learning_rate": {"min":0.00001, "max":0.0001, "prior":"log-uniform", "type":"float", "section":"general_param"},
			"reg_coef": {"min":0.001, "max":0.01, "prior":"log-uniform", "type":"float", "section":"general_param"},
			"kernel_size": {"structure":"[[[2, 8], [2, 8], [5, 100]], [[2, 6], [2, 6], [5, 100]], [[2, 4], [2, 4], [5, 100]]]", "type":"network", "section":"general_param"},
			"kernel_stride": {"structure":"[[[1, 2], [1, 2]], [[1, 2], [1, 2]], [[1, 2], [1, 2]]]", "type":"network", "section":"general_param"},
			"pool_size": {"structure":"[[[1, 2], [1, 2]], [[1, 2], [1, 2]], [[1, 2], [1, 2]]]", "type":"network", "section":"general_param"},
			"pool_stride": {"structure":"[[[1, 2], [1, 2]], [[1, 2], [1, 2]], [[1, 2], [1, 2]]]", "type":"network", "section":"general_param"},
			"fc_size": {"structure":"[[[16, 512]]]", "type":"network", "section":"general_param"},
			"gamma": {"min":0.6, "max":0.99, "prior":"uniform", "type":"float", "section":"rl_param"},
			"max_age": {"min":3, "max":10, "type":"int", "section":"rl_param"},
			"dbl_coef": {"min":0.01, "max":0.01, "prior":"uniform", "type":"float", "section":"rl_param"}
			}

constants={
			"train_path":{"value":"/dccstor/sentana/autodp/storage/inputs/dogcat/data50/tf_train", "section":"general_param"},
			"valid_path":{"value":"/dccstor/sentana/autodp/storage/inputs/dogcat/data50/tf_valid", "section":"general_param"},
			"test_path":{"value":"/dccstor/sentana/autodp/storage/inputs/dogcat/data50/tf_test", "section":"general_param"},
			"prep_path":{"value":"/dccstor/sentana/autodp/storage/inputs/dogcat/data50", "section":"general_param"},
			"result_path":{"value":"/dccstor/sentana/autodp/storage/results/dogcat", "section":"general_param"},
			"save_model":{"value":"/dccstor/sentana/autodp/storage/models/dogcat", "section":"general_param"},
			"ima_height":{"value":50, "section":"general_param"},
			"ima_width":{"value":50, "section":"general_param"},
			"ima_depth":{"value":1, "section":"general_param"},
			"num_class":{"value":2, "section":"general_param"},
			"num_epoch":{"value":300, "section":"general_param"},
			"batch_size":{"value":200, "section":"general_param"},
			"max_grad_norm":{"value":3, "section":"general_param"},
			"valid_step":{"value":200, "section":"general_param"},
			"reader":{"value":"autodp.reader.tf_reader.TFReader", "section":"general_param"},
			"nn_arch":{"value":"autodp.network.arch.nn.nn_arch.NNArch", "section":"nn_param"},
			"nn_loss":{"value":"autodp.network.loss.sce.SCE", "section":"nn_param"},
			"rl_arch":{"value":"autodp.network.arch.rl.dual_q_arch.DualQArch", "section":"rl_param"},
			"rl_loss":{"value":"autodp.network.loss.adv_log.AdvLog", "section":"rl_param"},
			"rl_graph":{"value":"autodp.network.graph.rl.policy_base.policy_graph.PolicyGraph", "section":"rl_param"},
			"rl_agent":{"value":"autodp.rl_core.agent.policy_base.gradient.baseline_reinforce.BaselineReinforce", "section":"rl_param"},
			"rl_action":{"value":"autodp.rl_core.env.action.cont_rotation.ContRotation", "section":"rl_param"},
			"num_action":{"value":13, "section":"rl_param"},
			"max_explore":{"value":1, "section":"rl_param"},
			"min_explore":{"value":0.1, "section":"rl_param"},
			"buf_size":{"value":40000, "section":"rl_param"},
			"anneal_step":{"value":100000, "section":"rl_param"}
			"kernel_size": {"value":[[7, 6, 1, 92], [5, 2, 92, 75], [2, 3, 75, 34]], "section":"general_param"},
			"kernel_stride": {"value":[[7, 6, 1, 92], [5, 2, 92, 75], [2, 3, 75, 34]], "section":"general_param"},
			"pool_size": {"value":[[7, 6, 1, 92], [5, 2, 92, 75], [2, 3, 75, 34]], "section":"general_param"},
			"pool_stride": {"value":[[7, 6, 1, 92], [5, 2, 92, 75], [2, 3, 75, 34]], "section":"general_param"},

			}









