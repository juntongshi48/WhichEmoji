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

from tokenizer import word_based

import pdb

"""
TODO:
1. Create Train/val/test datasets
2. Complete process()
"""


class twitter_dataset (data.Dataset):
    def __init__(self, data_path, tokenizer, batch_size=128, device="cpu", **kwargs):
        self.data_path = data_path
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.device = device
        self.kwargs = kwargs

        

        self.data = []
        self.process()      
    
    def process(self):
        df = pd.read_csv(self.data_path,  sep=",")

        labels = df.iloc[:,0].to_numpy(dtype=np.int32)
        sentences = df.iloc[:,1].to_numpy(dtype=str)

        encoded_sentences = tokenizer.process(sentences)
        pdb.set_trace()


        # sentences = [sentence.split('\t') for sentence in sentences]
        # filtered_sentence = []
        # for sentence in sentences:
        #     if len(sentence) >= self.min_sentence_len:
        #         filtered_sentence.append(sentence)

        # tokenizer.tokenize(filtered_sentence)
        # print(df[:10])
        # pdb.set_trace()

        # TODO: 
        # Tokenize Raw data
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# print(os.path.join(os.path, "src"))
train_path = "data/processed/train.csv"
tokenizer = word_based()
train_set = twitter_dataset(train_path, tokenizer)