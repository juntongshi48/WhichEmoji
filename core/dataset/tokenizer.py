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


class word_based():
    def __init__(self, unk_max_frequency=3, min_sentence_len=2, max_sentence_len=60) -> None:
        self.unk_max_frequency = unk_max_frequency
        self.min_sentence_len = min_sentence_len
        self.max_sentence_len = max_sentence_len
        self.vocab2id = {
            '<PAD>': 0,
            '<UNK>': 1
        }
        self.id2vocab = None

    def process(self, sentences, train):
        """ 
        Input: list of sentences separated by '\t'
        Output: list of tokenized and encoded sentences
        """
        tokenized_sentences = self.tokenize(sentences)
        if train: # vocab dictionary is computed only for the training set
            self.get_vocab_id(tokenized_sentences)
        encoded_sentences = self.encode(tokenized_sentences)
        return encoded_sentences

    def tokenize(self, sentences):
        ## Tokenize
        tokenized_sentences = [sentence.split('\t') for sentence in sentences]
        ## Filter: drop short sentences
        filtered = []
        for sentence in tokenized_sentences:
            if len(sentence) >= self.min_sentence_len:
                filtered.append(sentence)
        tokenized_sentences = filtered
        return tokenized_sentences
    
    def encode(self, tokenized_sentences):
        # sentence_len = [len(sentence) for sentence in tokenized_sentences]
        # print(max(sentence_len), min(sentence_len))
        # print(sum(sentence_len)/len(sentence_len))
        # pdb.set_trace()
        encoded_sentences_with_eos = [self.encode_and_pad_sentence(tokenized_sentence) for tokenized_sentence in tokenized_sentences]
        encoded_sentences, eos = zip(*encoded_sentences_with_eos)
        return list(encoded_sentences), list(eos) 

    def encode_and_pad_sentence(self, tokenized_sentence):
        # Input: list of tokens
        # Output: Tensor of embeddings
        ## Encode
        embedding = [self.vocab2id.get(t, self.vocab2id['<UNK>']) for t in tokenized_sentence]
        ## Padding
        eos = len(embedding)-1
        if len(embedding) > self.max_sentence_len:
            embedding = embedding[:self.max_sentence_len]
            eos = self.max_sentence_len - 1
        elif len(embedding) < self.max_sentence_len:
            embedding += [self.vocab2id['<PAD>']] * (self.max_sentence_len - len(embedding))
        return embedding, eos

    def get_vocab_id(self, sentences):
        corpus = [word for sentence in sentences for word in sentence]
        vocab2count = Counter(corpus)

        next_v_id = len(self.vocab2id)
        for v in vocab2count:
            if vocab2count[v] < 3:  # anything occuring less than 3 times will be replaced by <UNK>
                continue
            elif v not in self.vocab2id:
                self.vocab2id[v] = next_v_id
                next_v_id += 1
        self.id2vocab = {v: k for k, v in self.vocab2id.items()}
