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

class multilabel_dataset(data.Dataset):
    def __init__(self, data_path, tokenizer, train=False, device="cpu", **kwargs):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.train = train
        self.device = device
        self.kwargs = kwargs

        self.encoded_sentences = None
        self.labels = None
        self.process()

    def process(self):
        df = pd.read_csv(self.data_path, sep=",")
        # Convert the string representation of labels to a list of integers
        self.labels = [self._convert_label(label) for label in df.iloc[:, 0]]
        sentences = df.iloc[:, 1].to_numpy(dtype=str)
        self.encoded_sentences, self.eos = self.tokenizer.process(sentences, self.train)

    def _convert_label(self, label_str):
        # Remove the brackets and convert the string to a list of integers
        label_list = [int(i) for i in label_str.strip('[]').split(',')]
        return label_list

    def __len__(self):
        return len(self.encoded_sentences)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded_sentences[idx], dtype=torch.long, device=self.device)
        eos = torch.tensor(self.eos[idx], dtype=torch.long, device=self.device)
        y = torch.tensor(self.labels[idx], dtype=torch.float, device=self.device)  # Use float for binary vector
        return x, eos, y
