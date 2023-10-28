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

vocab2id = {
    '<PAD>': 0,
    '<UNK>': 1,
    '<s>': 2,
    '</s>': 3
}

class twitter_dataset (data.Dataset):
    def __init__(self, data_path,  batch_size, tokenizer, device="cpu", **kwargs):
        self.data_path = data_path
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.device = device
        self.kwargs = kwargs

        

        self.data = []
        self.process()      
    
    def process(self):
        df = pd.read_csv(self.data_path,  sep=",")
        ds = df.to_numpy()
        labels = ds[:,0]
        sentences = ds[:,1]
        
        
        # print(df[:10])
        # pdb.set_trace()

        # TODO: 
        # Tokenize Raw data
        return

    def word_based(self, sentences):
        corpus = '\t'.join(sentences)
        vocab2count = Counter(sentences.split('\t'))

        next_v_id = 4
        for v in vocab2count:
            if vocab2count[v] < 3:  # anything occuring less than 3 times will be replaced by <UNK>
                continue
            elif v not in vocab2id:  # <s> and </s> already in vocab2id
                vocab2id[v] = next_v_id
                next_v_id += 1

        id2vocab = {v: k for k, v in vocab2id.items()}
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# print(os.path.join(os.path, "src"))
train_path = "core/dataset/data/processed/train.csv"
train_set = twitter_dataset(train_path, 128)