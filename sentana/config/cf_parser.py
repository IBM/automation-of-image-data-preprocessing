"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
import os

import configparser


PATH_TO_CONFIG_FILE = os.path.join(os.path.dirname(__file__),
                                   "../../etc/sentana.cfg" )


# Check if file exists and readable
if not (os.path.isfile(PATH_TO_CONFIG_FILE) and os.access(
        PATH_TO_CONFIG_FILE, os.R_OK)):
    raise IOError("Either file %s is missing or not "
                  "readable." % PATH_TO_CONFIG_FILE)

# Create a configuration parser to parse all necessary parameters
config_parser = configparser.RawConfigParser()
config_parser.read(PATH_TO_CONFIG_FILE)
