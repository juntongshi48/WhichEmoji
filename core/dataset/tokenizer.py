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

import transformers

import pdb


class WordBasedTokenizer():
    def __init__(self, unk_max_frequency=3, min_sentence_len=2, max_sentence_len=60) -> None:
        self.unk_max_frequency = unk_max_frequency
        self.min_sentence_len = min_sentence_len
        self.max_sentence_len = max_sentence_len
        self.vocab2id = {
            '<PAD>': 0,
            '<UNK>': 1
        }
        self.id2vocab = None
        self.above_max = 0

    def process(self, sentences, train):
        """ 
        Input: a list of sentences with words separated by '\t'
        Output: a list of tokenized and encoded sentences
        """
        tokenized_sentences = self.tokenize(sentences)
        if train: # vocab dictionary is computed only for the training set
            self.get_vocab_id(tokenized_sentences)
        encoded_sentences, eos = self.encode(tokenized_sentences)
        return encoded_sentences, eos

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
        print(f"The proportion of tweets that got truncateddue to exceeding the maximum lenght of {self.max_sentence_len} is: {100*self.above_max/len(tokenized_sentences)}%")
        self.above_max = 0
        return list(encoded_sentences), list(eos) 

    def encode_and_pad_sentence(self, tokenized_sentence):
        # Input: list of tokens
        # Output: Tensor of embeddings
        ## Encode
        embedding = [self.vocab2id.get(t, self.vocab2id['<UNK>']) for t in tokenized_sentence]
        ## Padding
        eos = len(embedding)-1
        if len(embedding) > self.max_sentence_len:
            self.above_max += 1
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
            if vocab2count[v] < self.unk_max_frequency:  # anything occuring less than 3 times will be replaced by <UNK>
                continue
            elif v not in self.vocab2id:
                self.vocab2id[v] = next_v_id
                next_v_id += 1
        self.id2vocab = {v: k for k, v in self.vocab2id.items()}

class PretrainedTokenizer:
    def __init__(self, name: str = 'gpt2', unk_max_frequency=3, min_sentence_len=2, max_sentence_len=200) -> None:
        self.min_sentence_len = min_sentence_len
        self.max_sentence_len = max_sentence_len
        self.unk_max_frequency = unk_max_frequency
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        # Set padding token
        # Note: we also use the padding token as UNK token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pretrained_dict = self.tokenizer.get_vocab()
        self.vocab2id = {
            '<PAD>': 0,
            '<UNK>': 1
        }
        self.id2vocab = None

    # def process(self, sentences, train):
    #     """ 
    #     Input: a list of sentences with words separated by '\t'
    #     Output: a list of tokenized and encoded sentences
    #     """
    #     self.vocab2id = self.tokenizer.get_vocab()
    #     tokenized_sentences_ = []
    #     encoded_sentences_ = []
    #     eos_ = []
    #     for sent in tqdm(sentences):
    #         ## Tokenize
    #         sent = sent.replace('\t', ' ')
    #         tokenized_sent = self.tokenizer.tokenize(sent)
    #         ## Encode
    #         token2idx = lambda token: self.vocab2id[token]
    #         encoded_sent = list(map(token2idx, tokenized_sent))
    #         ## Padding
    #         eos = len(encoded_sent) - 1
    #         if len(encoded_sent) > self.max_sentence_len:
    #             encoded_sent = encoded_sent[:self.max_sentence_len]
    #             eos = self.max_sentence_len - 1
    #         elif len(encoded_sent) < self.max_sentence_len:
    #             idx_padding = self.vocab2id[self.tokenizer.pad_token]
    #             encoded_sent += [idx_padding] * (self.max_sentence_len - len(encoded_sent))
            
    #         tokenized_sentences_.append(tokenized_sent)
    #         encoded_sentences_.append(encoded_sent)
    #         eos_.append(eos)

    #     corpus = [word for sent in tokenized_sentences_ for word in sent]
    #     vocab2count = Counter(corpus)
    #     print(type(vocab2count))
    #     cnt = 0
    #     for v in vocab2count:
    #         if vocab2count[v] < 5:
    #             continue
    #         else:
    #             cnt += 1
    #     print(cnt)
    #     pdb.set_trace()
    #     return encoded_sentences_, eos_
    def process(self, sentences, train):
        """ 
        Input: a list of sentences with words separated by '\t'
        Output: a list of tokenized and encoded sentences
        """
        tokenized_sentences = self.tokenize(sentences)
        if train: # vocab dictionary is computed only for the training set
            self.get_vocab_id(tokenized_sentences)
        encoded_sentences, eos = self.encode(tokenized_sentences)
        return encoded_sentences, eos
    
    def tokenize(self, sentences):
        tokenized_sentences = [sent.replace('\t', ' ') for sent in sentences]
        ## Filter: drop short sentences
        filtered = []
        for sentence in tokenized_sentences:
            if len(sentence) >= self.min_sentence_len:
                filtered.append(sentence)
        ## Tokenize
        tokenized_sentences = [self.tokenizer.tokenize(sent) for sent in filtered]
        return tokenized_sentences
    
    def encode(self, tokenized_sentences):
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
            if vocab2count[v] < self.unk_max_frequency:  # anything occuring less than 3 times will be replaced by <UNK>
                continue
            elif v not in self.vocab2id:
                self.vocab2id[v] = next_v_id
                next_v_id += 1
        self.id2vocab = {v: k for k, v in self.vocab2id.items()}