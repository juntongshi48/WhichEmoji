import yaml
import os
import re
import fire
from ast import literal_eval
import argparse
import json
import copy
from dataclasses import dataclass

import pdb

@dataclass(frozen=True)
class Struct:
    __slots__ = ["__dict__"]
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            # recursively convert to struct
            if isinstance(value, dict):
                self.__dict__[key] = Struct(**value)
            else:
                self.__dict__[key] = value

    def todict(self):
        # recursively convert to dict
        return {
            k: v.todict() if isinstance(v, Struct) else v
            for k, v in self.__dict__.items()
        }

    def __getitem__(self, index):
        return self.__dict__[index]

@dataclass(frozen=True)
class Config:
    __slots__ = ["__dict__"]
    def __init__(self, config_file, **kwargs):
        _config = parse_config_file(path=config_file)
        _config.update(kwargs) # merge config_file and sub_arguments
        for key, value in _config.items():
            if isinstance(value, dict):
                self.__dict__[key] = Struct(**value)
            else:
                self.__dict__[key] = value

    def __getitem__(self, index):
        return self.__dict__[index]

    def todict(self):
        # recursively convert to dict
        return {
            k: v.todict() if isinstance(v, Struct) else v
            for k, v in self.__dict__.items()
        }

def parse_config_file(
    path=None,
    data=None,
    subs_dict={},
):
    loader = yaml.FullLoader
    if path:
        with open(path) as conf_data:
            return yaml.load(conf_data, Loader=loader)
    elif data:
        return yaml.load(data, Loader=loader)
    else:
        raise ValueError("Either a path or data should be defined as input")
