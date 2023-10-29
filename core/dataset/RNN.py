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

class RNNLM(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.vocab_size = params['vocab_size']
        self.d_emb = params['d_emb']
        self.d_hid = params['d_hid']
        self.n_layer = 1
        self.batch_size = params['batch_size']
        self.device = params['device']
        self.num_class = params['num_class']

        self.encoder = nn.Embedding(self.vocab_size, self.d_emb)
        self.rnn = nn.RNN(self.d_emb, self.d_hid, self.n_layer, batch_first=True)
        self.decoder = nn.Linear(self.d_hid, self.num_class)

    def forward(self, x, eos):
        """
            Inputs
            x = (N, L)
            eos = (N,)
        """
        batch_size, seq_len= x.shape
        hidden = (torch.zeros(self.n_layer, batch_size, self.d_hid).to(self.device))  # initial hidden state set to zeros
        ## Pass words through the embedding layer
        x = self.encoder(x) # (N, L, H_in)
        # x = x.transpose(-2, -1) # (N, H_in, L)
        ## Pass x, h0 into the RNN
        out, z_L = self.rnn(x, hidden) 
        z = out[:, eos, :] # (N, H_out)
        logit = self.decoder(z) # (N, K)
        return logit
    
    def loss(self, x, eos, y):
        """
            Inputs
            x = (N, L)
            eos = (N,)
            y = (N,)
        """
        n = x.shape[0]
        y_pred = self.forward(x, eos) # (N,K)
        loss = nn.CrossEntropyLoss(y_pred, y)
        avg_loss = loss / n
        return avg_loss



class ATTNLM(nn.Module):
    def __init__(self, params):
        super(ATTNLM, self).__init__()

        self.vocab_size = params['vocab_size']
        self.d_emb = params['d_emb']
        self.d_hid = params['d_hid']
        self.n_layer = 1
        self.btz = params['batch_size']

        self.encoder = nn.Embedding(self.vocab_size, self.d_emb)
        self.attn = Attention(self.d_hid)
        self.rnn = nn.RNN(self.d_emb, self.d_hid, self.n_layer, batch_first=True)
        # the combined_W maps to map combined hidden states and context vectors to d_hid
        self.combined_W = nn.Linear(self.d_hid * 2, self.d_hid)
        self.decoder = nn.Linear(self.d_hid, self.vocab_size)


    def forward(self, batch, return_attn_weights=False):

        """
            IMPLEMENT HERE
            Copy your implementation of RNNLM, make sure it passes the RNNLM check
            In addition to that, you need to add the following 3 things
            1. pass rnn output to attention module, get context vectors and attention weights
            2. concatenate the context vec and rnn output, pass the combined
               vector to the layer dealing with the combined vectors (self.combined_W)
            3. if return_attn_weights, instead of returning the [N, L, V]
               matrix, return the attention weight matrix of dimension [N, L, L]
               which was received from the forward function of attnetion module
        """
        batch_size, seq_len= batch.shape
        hidden = torch.zeros(self.n_layer, batch_size, self.d_hid).to(device)
        ## Pass words through the embedding layer
        x = self.encoder(batch) # (N, L, H_in)
        # x = x.transpose(-2, -1) # (N, H_in, L)
        ## Pass x, h0 into the RNN
        rnn_out = self.rnn(x, hidden) 
        z = rnn_out[0] # (N, L, H_out)

        att_vec, att_scores = self.attn(z)
        decoder_input = self.combined_W(torch.cat((att_vec, z), dim=-1))
        logit = self.decoder(decoder_input) # (N, L, V)
        
        if return_attn_weights:
            return att_scores
        else:
            return logit


class Attention(nn.Module):
    def __init__(self, d_hidden):
        super(Attention, self).__init__()
        self.d_hid = d_hidden


    def forward(self, x):

        """
            IMPLEMENT HERE
            For each time step t
                1. Obtain attention scores for step 0 to (t-1)
                   This should be a dot product between current hidden state (x[:,t:t+1,:])
                   and all previous states x[:, :t, :]. While t=0, since there is not
                   previous context, the context vector and attention weights should be of zeros.
                   You might find torch.bmm useful for computing over the whole batch.
                2. Turn the scores you get for 0 to (t-1) steps to a distribution.
                   You might find F.softmax to be helpful.
                3. Obtain the sum of hidden states weighted by the attention distribution
            Concat the context vector you get in step 3. to a matrix.

            Also remember to store the attention weights, the attention matrix for
            each training instance should be a lower triangular matrix. Specifically, in
            each row, element 0 to t-1 should sum to 1, the rest should be padded with 0.
            e.g.
            [ [0.0000, 0.0000, 0.0000, 0.0000],
              [1.0000, 0.0000, 0.0000, 0.0000],
              [0.4246, 0.5754, 0.0000, 0.0000],
              [0.2798, 0.3792, 0.3409, 0.0000] ]

            Return the context vector matrix and the attention weight matrix

        """
        batch_seq_len = x.shape[1]
        prev_states = x
        curr_states = x.transpose(-1,-2)
        att_scores = torch.bmm(prev_states, curr_states) # (N, L, L)
        ## Mask out h_{t_1} * h_{t_2} when t_1 >= t_2
        att_scores = torch.tril(att_scores, diagonal=-1)
        ## BUG: the raw logits should be masked into negative infinity rather than 0
        neg_infty = -1e9
        neg_infty = torch.ones_like(att_scores) * neg_infty
        neg_infty = torch.triu(neg_infty)
        att_scores += neg_infty
        ## Normalize
        att_scores = F.softmax(att_scores, dim=-1) # Take the softmax over the last dimension
        
        ## Compute the context vector matrix
        att_vec = torch.bmm(att_scores, x)
        return att_vec, att_scores