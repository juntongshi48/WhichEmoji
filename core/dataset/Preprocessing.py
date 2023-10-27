import sys 
import os
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import train_test_split

class preprocessing:
    def __init__(self, data_paths) -> None:
        self.data_paths = data_paths
    def read_csv(data_paths):
        



data_paths = {1: "cooking", 2: "sun", 3:"clown_face"}
for key in data_paths:
    data_paths[key] = "data/processed_data" + data_paths[key] + ".cvs"