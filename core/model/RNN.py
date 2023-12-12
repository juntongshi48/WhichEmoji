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
import torch.nn.functional as F

import pdb

class SoftMaxLoss(nn.Module):
    """
        logit: (N,C), raw logits
        y: (N,C), values should be {0,1}
        pos_weight: trade-off between the loss of positive and negative classes
            pos_weight > 1 favors recall, as those ground-truth positve samples are increasingly weighted
            pos_weight < 1 favors precision.
        L(logit, y) = -(pos_weight*y*log(softmax(logit) + (1-y)*log(1-softmax(logit)))
    """
    def __init__(self, pos_weight: float=1):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, logit, y):
        # Normalize multicalss label y to probabilities
        num_pos_class = torch.sum(y, dim=1, keepdim=True)
        y = y / torch.sum(y, dim=1, keepdim=True)
        # Convert logits to probabilities
        epsilon = 1e-8
        prob = F.softmax(logit, dim=1)
        prob = (prob + epsilon) / (1 + epsilon * logit.shape[1]) # for numerical stability, avoid zero prob
        # Compute loss
        total_loss = - (self.pos_weight * y * torch.log(prob) + (1-y) * torch.log(1-prob))
        avg_loss = torch.mean(total_loss)
        return avg_loss

class RNNLM(nn.Module):
    def __init__(self, cfg, num_class, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_embedding = cfg.model.d_embedding
        self.d_hidden = cfg.model.d_hidden
        self.n_layer = cfg.model.n_layer
        self.batch_size = cfg.batch_size
        self.device = cfg.device
        self.num_class = num_class

        self.encoder = nn.Embedding(self.vocab_size, self.d_embedding)
        self.rnn = nn.RNN(self.d_embedding, self.d_hidden, self.n_layer, batch_first=True)
        self.decoder = nn.Linear(self.d_hidden, self.num_class)
        
        self.num_output_labels = cfg.num_output_labels
        self.loss_func = nn.CrossEntropyLoss()
        if self.num_output_labels > 1:
            if cfg.model.loss == "softmax":
                self.loss_func = SoftMaxLoss(cfg.model.pos_weight)
            elif cfg.model.loss == "BCE":
                self.loss_func = nn.BCEWithLogitsLoss()
            else:
                raise NotImplementedError("The loss funcion {cfg.model.loss} is not implemented")
        # self.epoch = 0
        self.threshold = cfg.model.threshold if hasattr(cfg.model, "threshold") else (1/self.num_class) 

    def forward(self, x, eos):
        """
            Inputs
            x = (N, L)
            eos = (N,)
        """
        batch_size, seq_len= x.shape
        hidden = (torch.zeros(self.n_layer, batch_size, self.d_hidden).to(self.device))  # initial hidden state set to zeros
        ## Pass words through the embedding layer
        x = self.encoder(x) # (N, L, H_in)
        ## Pass x, h0 into the RNN
        out, z_L = self.rnn(x, hidden) # out: (N,L,H_out)
        eos = F.one_hot(eos, num_classes=seq_len) # (N,L)
        eos = eos.to(torch.float32) # BUG: matmul doesn't support int
        z = torch.matmul(out.transpose(-1,-2), eos.unsqueeze(-1)) # (N, H_out) BUG: under torch matmul/bmm, I need to unsqueeze the column vector eos
        z = z.squeeze(-1)
        logit = self.decoder(z) # (N, K)

        return logit
    
    def loss(self, x, eos, y, epoch):
        """
            Inputs
            x = (N, L)
            eos = (N,)
            y = (N,)
        """
        n = x.shape[0]
        logit = self.forward(x, eos) # (N,K)
        if self.num_output_labels > 1:
            y = y.to(torch.float32) # BUG: nn.BCEWithLogitsLoss() requires the target to be float values
        loss = self.loss_func(logit, y) # BUG: nn.CrossEntropyLoss() by default takes the mean (recall the argument reduction='mean')
        # For DEBUG -- compare logits with true labels
        # if self.epoch == 1:
        #     torch.set_printoptions(threshold=10_000)
        #     print(torch.cat((logit,y.unsqueeze(-1)), dim=-1)[:10])
        #     print(f'loss: {loss}')
        #     pdb.set_trace()
        
        # if epoch == 10:
        #     l1 = self.loss_func(logit[:2], y[:2])
        #     l2 = self.loss_func(y[:2]*100, y[:2])
        #     l3 = self.loss_func((y[:2]*2-1)*100, y[:2])
        #     print(logit[:2])
        #     print(l1)
        #     print(y[:2]*100)
        #     print(l2)
        #     print((y[:2]*2-1)*100)
        #     print(l3)
        #     pdb.set_trace()
        return loss

    def predict(self, x, eos, recommand=False):
        """
            if single-label
                output: [N], each row contains the predicted label
            if multi-label
                if not recommand
                    output: [N,C], each row contains a 10-class prediction 
                        determined by the prediced probabilitie and the threshold
                if recommand
                    output: [N, num_output_labels], each row contains the top recommanded labels
        """
        logit = self.forward(x, eos) # (N,K)
        with torch.no_grad():
            y_pred = torch.argsort(logit, dim=1, descending=True)[:,:self.num_output_labels]
            y_pred = torch.squeeze(y_pred) # squeeze for single-class
            if self.num_output_labels > 1 and not recommand:
                prob = F.softmax(logit, dim=1)
                y_pred = prob > self.threshold
            return y_pred.cpu().numpy()
        

class ATTNLM(RNNLM):
    def __init__(self, cfg, num_class, vocab_size):
        super().__init__(cfg, num_class, vocab_size)
        self.attn = Attention()
        self.rnn = nn.RNN(self.d_embedding, self.d_hidden, self.n_layer, batch_first=True)
        # the combined_W maps to map combined hidden states and context vectors to d_hidden
        self.combined_W = nn.Linear(self.d_hidden * 2, self.d_hidden)
        self.decoder = nn.Linear(self.d_hidden, self.num_class)


    def forward(self, x, eos, return_attn_weights=False):

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
        batch_size, seq_len= x.shape
        hidden = (torch.zeros(self.n_layer, batch_size, self.d_hidden).to(self.device))  # initial hidden state set to zeros
        ## Pass words through the embedding layer
        x = self.encoder(x) # (N, L, H_in)
        ## Pass x, h0 into the RNN
        out, z_L = self.rnn(x, hidden) # out: (N,L,H_out)
        eos = F.one_hot(eos, num_classes=seq_len) # (N,L)
        eos = eos.to(torch.float32) # BUG: matmul doesn't support int
        z = torch.matmul(out.transpose(-1,-2), eos.unsqueeze(-1)) # (N, H_out) BUG: under torch matmul/bmm, I need to unsqueeze the column vector eos
        z = z.squeeze(-1)

        att_vec, att_scores = self.attn(out)
        att_z = torch.matmul(att_vec.transpose(-1,-2), eos.unsqueeze(-1)) # (N, H_out) BUG: under torch matmul/bmm, I need to unsqueeze the column vector eos
        att_z = att_z.squeeze(-1)
        decoder_input = self.combined_W(torch.cat((att_z, z), dim=-1))
        logit = self.decoder(decoder_input) # (N, K)
        
        if return_attn_weights:
            return att_scores
        else:
            return logit


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        # self.d_hidden = d_hidden


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