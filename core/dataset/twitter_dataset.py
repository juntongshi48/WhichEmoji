import sys 
import os
from collections import OrderedDict, Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

import pdb

"""
TODO:
1. Create Train/val/test datasets
2. Complete process()
"""

class twitter_dataset (data.Dataset):
    def __init__(self, data_path, tokenizer, train=False, device="cpu", **kwargs):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.train = train
        self.device = device
        self.kwargs = kwargs

        # self.data = None
        self.encoded_sentences = None
        self.labels = None
        self.process()      
    
    def process(self):
        df = pd.read_csv(self.data_path,  sep=",")
        self.labels = df.iloc[:,0].to_numpy(dtype=np.int32)
        sentences = df.iloc[:,1].to_numpy(dtype=str)
        self.encoded_sentences, self.eos = self.tokenizer.process(sentences, self.train)
        # self.data = zip(encoded_sentences, labels)

        # xytuple = zip(encoded_sentences, labels)
        # tuple2pair = lambda xytuple: xypair(*xytuple)
        # self.data = list(map(tuple2pair, xytuple))

    def __len__(self):
        return len(self.encoded_sentences)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded_sentences[idx], dtype=torch.long)
        eos = torch.tensor (self.eos[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, eos, y


# # print(os.path.join(os.path, "src"))
# train_path = "core/dataset/data/processed/train.csv"
# tokenizer = word_based()
# train_set = twitter_dataset(train_path, tokenizer)