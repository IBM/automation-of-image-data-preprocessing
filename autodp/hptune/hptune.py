"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os

import configparser
import json
import ast
from skopt import gp_minimize
import numpy as np
import tensorflow as tf

cfg_file = "/Users/minhtn/ibm/projects/autodp/configs/hptune.cfg"


class HPTune(object):
    """This class implements an hyper-optimizer for hyper-parameter tuning."""
    def __init__(self):
        """Initialization by creating a search space."""
        self._space = self._create_space()
        self._count = 0

    def run_tuning(self, n_call=10):
        """Main method to run hyper tuning."""
        res = gp_minimize(self._obj_fun, self._space, acq_func="EI", n_calls=n_call, verbose=True)
        self._write_config(res.x)
        from autodp import cf
        cf.reset_config(self._config_input.get("config", "output"))
        return res

    def _create_space(self):
        """Function to create a search space."""
        # Check hyper config file
        if not os.path.isfile(cfg_file):
            raise FileNotFoundError("HPTune config file not exist at ", os.path.abspath(cfg_file))

        # Setup a config parser
        self._config_input = configparser.RawConfigParser()
        self._config_input.read(cfg_file)

        # Create search space
        space = []
        self._search_space = json.loads(self._config_input.get("config", "search_space"))
        for hp in np.unique(list(self._search_space.keys())):
            hp = self._search_space[hp]
            if hp["type"] == "float":
                if "prior" in hp:
                    space.append((hp["min"], hp["max"], hp["prior"]))
                else:
                    space.append((hp["min"], hp["max"]))

            elif hp["type"] == "int":
                if "prior" in hp:
                    space.append((int(hp["min"]), int(hp["max"]), hp["prior"]))
                else:
                    space.append((int(hp["min"]), int(hp["max"])))

            elif hp["type"] == "network":
                structure = ast.literal_eval(hp["structure"])
                for l in structure:
                    for minimum, maximum in l:
                        space.append((minimum, maximum))
        return space

    def _write_config(self, x):
        """Write config file."""
        # Initialize a config object
        config = configparser.RawConfigParser()

        # Compute the config object
        i = 0
        for hp in np.unique(list(self._search_space.keys())):
            # Add section if not available yet
            if not config.has_section(self._search_space[hp]["section"]):
                config.add_section(self._search_space[hp]["section"])

            # Add value for hyper parameters
            if self._search_space[hp]["type"] == "int" or self._search_space[hp]["type"] == "float":
                config.set(self._search_space[hp]["section"], hp, x[i])
                i += 1
            elif self._search_space[hp]["type"] == "network":
                struct = ast.literal_eval(self._search_space[hp]["structure"])
                value, fix_value = [], int(self._config_input.get("config", "ima_depth"))
                for j in range(len(struct)):
                    if hp == "fc_size":
                        value.append(x[i] - x[i] % 2)
                        i += 1
                    elif hp == "kernel_size":
                        value.append([x[i], x[i+1], fix_value, x[i+2]])
                        fix_value = x[i+2]
                        i += 3
                    else:
                        value.append([1, x[i], x[i+1], 1])
                        i += 2
                config.set(self._search_space[hp]["section"], hp, value)

        # Add constants to the config object
        constants = json.loads(self._config_input.get("config", "constants"))
        for e in constants:
            if not config.has_section(constants[e]["section"]):
                config.add_section(constants[e]["section"])
            config.set(constants[e]["section"], e, constants[e]["value"])

        # Write config file
        with open(self._config_input.get("config", "output"), "w") as autodp_cf:
            config.write(autodp_cf)

        with open(self._config_input.get("config", "output") + str(self._count),
                  "w") as autodp_cf:
            config.write(autodp_cf)
        self._count += 1

    def _obj_fun(self, x):
        """The objective function for hyper tuning, i.e., performance of the validation set."""
        # Write configs for autodp
        self._write_config(x)

        # Read new configs
        from autodp import cf
        cf.reset_config(self._config_input.get("config", "output"))

        # Reset default graph
        tf.reset_default_graph()

        # Run the runner to get objective value
        m = __import__(self._config_input.get("objective", "module"), self._config_input.get("objective", "function"))
        objective = getattr(m, self._config_input.get("objective", "function"))
        return objective().train_model(verbose=False)
