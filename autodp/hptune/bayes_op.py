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

cfg_file = "/Users/minhtn/ibm/projects/One/onebm_afl/etc/hyperopt.cfg"


class HyperOpt(object):
    def __init__(self):
        self.space, hp_names = self.create_space()
        print("Param is:", hp_names)
        print("Space is:", self.space)

    def run_tuning(self):
        res = gp_minimize(self.obj_fun, self.space, acq_func="EI",
                          n_calls=10, verbose=True)
        print(res)

        return res

    def create_space(self):
        if os.path.isfile(cfg_file):
            self.config_input = configparser.RawConfigParser()
            self.config_input.read(cfg_file)
            space = []
            self.search_space = json.loads(self.config_input.get("config",
                                                                 "search_space"))
            self.schemas = json.loads(
                self.config_input.get("config", "schemas"))
            self.cell_sizes = json.loads(self.config_input.get("config",
                                                               "cell_sizes"))
            hp_names = []
            for hp in np.unique(list(self.search_space.keys())):
                hp_names.append(hp)
                hp = self.search_space[hp]
                if hp["type"] == "int" or hp["type"] == "float":
                    if "prior" in hp:
                        space.append((hp["min"], hp["max"], hp["prior"]))

                    else:
                        space.append((hp["min"], hp["max"]))

                elif hp["type"] == "schema":
                    index = ast.literal_eval(hp["index"])
                    space.append((0, int(np.log(ast.literal_eval(
                        self.schemas[hp["ref"]]["value"])[index[0]][
                                                    index[1]][0]))))

                elif hp["type"] == "cell_size":
                    space.append((hp["min"], hp["max"]))

                elif hp["type"] == "network":
                    structure = ast.literal_eval(hp["structure"])
                    for l in structure:
                        for minimum, maximum in l:
                            space.append((minimum, maximum))

            return space, hp_names
        else:
            raise FileNotFoundError("Hyperopt config file does not exist at",
                                    os.path.abspath(cfg_file))

    def write_config(self, x, tune=True):
        """ Write config file """
        config = configparser.RawConfigParser()

        i = 0
        for hp in np.unique(list(self.search_space.keys())):
            if self.search_space[hp]["type"] == "int" \
                    or self.search_space[hp]["type"] == "float":
                if not config.has_section(self.search_space[hp]["section"]):
                    config.add_section(self.search_space[hp]["section"])
                config.set(self.search_space[hp]["section"], hp, x[i])
                i += 1
            elif self.search_space[hp]["type"] == "schema":
                if not config.has_section(
                        self.schemas[self.search_space[hp]["ref"]]["section"]):
                    config.add_section(
                        self.schemas[self.search_space[hp]["ref"]]["section"])
                index = ast.literal_eval(self.search_space[hp]["index"])
                if not config.has_option(
                        self.schemas[self.search_space[hp]["ref"]]["section"],
                        self.search_space[hp]["ref"]):
                    config.set(
                        self.schemas[self.search_space[hp]["ref"]]["section"],
                        self.search_space[hp]["ref"],
                        self.schemas[self.search_space[hp]["ref"]]["value"])
                schema = ast.literal_eval(config.get(
                    self.schemas[self.search_space[hp]["ref"]]["section"],
                    self.search_space[hp]["ref"]))
                if len(schema[index[0]][index[1]]) == 3:
                    schema[index[0]][index[1]] = (
                        schema[index[0]][index[1]][0], x[i],
                        schema[index[0]][index[1]][2])
                elif len(schema[index[0]][index[1]]) == 2:
                    schema[index[0]][index[1]] = (
                        schema[index[0]][index[1]][0], x[i])
                config.set(
                    self.schemas[self.search_space[hp]["ref"]]["section"],
                    self.search_space[hp]["ref"], str(schema))
                i += 1

            elif self.search_space[hp]["type"] == "cell_size":
                if not config.has_section(
                        self.cell_sizes[self.search_space[hp]["ref"]][
                            "section"]):
                    config.add_section(
                        self.cell_sizes[self.search_space[hp]["ref"]][
                            "section"])
                if not config.has_option(
                        self.cell_sizes[self.search_space[hp]["ref"]][
                            "section"], self.search_space[hp]["ref"]):
                    config.set(self.cell_sizes[self.search_space[hp]["ref"]][
                                   "section"], self.search_space[hp]["ref"],
                               self.cell_sizes[self.search_space[hp]["ref"]][
                                   "value"])
                cell_size = ast.literal_eval(config.get(
                    self.cell_sizes[self.search_space[hp]["ref"]]["section"],
                    self.search_space[hp]["ref"]))
                cell_size[int(self.search_space[hp]["index"])] = x[i]
                config.set(
                    self.cell_sizes[self.search_space[hp]["ref"]]["section"],
                    self.search_space[hp]["ref"], str(cell_size))
                i += 1

            elif self.search_space[hp]["type"] == "network":
                if not config.has_section(self.search_space[hp]["section"]):
                    config.add_section(self.search_space[hp]["section"])
                structure = ast.literal_eval(self.search_space[hp]["structure"])
                for j in range(len(structure)):
                    previous_is_zero = False
                    for k in range(len(structure[j])):
                        if x[i] == 0 or previous_is_zero:
                            structure[j].pop()
                            previous_is_zero = True
                        else:
                            structure[j][k] = x[i]
                        i += 1
                if hp == "layers":
                    config.set(self.search_space[hp]["section"], hp,
                               str(structure[0]))
                else:
                    config.set(self.search_space[hp]["section"], hp,
                               str(structure))

        constants = json.loads(self.config_input.get("config", "constants"))
        for e in constants:
            if not config.has_section(constants[e]["section"]):
                config.add_section(constants[e]["section"])
            config.set(constants[e]["section"], e, constants[e]["value"])

        if not tune:
            train_path = config.get("general_param", "train_path") + "_full"
            valid_path = config.get("general_param", "valid_path") + "_full"
            config.set("general_param", "train_path", train_path)
            config.set("general_param", "valid_path", valid_path)

        with open(self.config_input.get("config", "output"), 'w') as configfile:
            config.write(configfile)

    def obj_fun(self, x, tune=True, save=False):
        self.write_config(x, tune=tune)

        module = __import__(self.config_input.get("objective", "module"),
                            fromlist=self.config_input.get("objective",
                                                           "function"))
        objective = getattr(module, self.config_input.get("objective",
                                                          "function"))

        return objective().hyper_train(
            save=save, config=self.config_input.get("config", "output"))





