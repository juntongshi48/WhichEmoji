import sys 
import os
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

"""
TODO:
1. Create Train/val/test datasets
2. Complete process()
"""

class Twitters (data.Dataset):
    def __init__(self, data_path,  batch_size, device= "cpu", **kwargs):
        self.data_path = data_path
        self.batch_size = batch_size
        self.device = device
        self.kwargs = kwargs

        self.data = []
        self.process(self.data_path)      
    
    def process(data_path):
        
        # TODO: 
        # read train/val/teset files with CVS reader and convert them into self.data
        # OR find some existing code
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# print(os.path.join(os.path, "src"))
# data_path = {1: "cooking", 2: "sun", 3:"clown_face"}
# for key in data_path:
#     data_path[key] = "data/processed_data" + data_path[key] + ".cvs"