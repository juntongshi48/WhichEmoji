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

from tokenizer import word_based
from twitter_dataset import twitter_dataset
from RNN import RNNLM

def train(model, train_loader, optimizer, epoch, grad_clip=None, rectify=False):
    model.train()
    losses = []
    for x, eos, y in train_loader:
        pdb.set_trace()
        x = x.cuda()
        eos = eos.cuda()
        y = y.cuda()
        loss = model.loss(x, eos, y)
        
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # desc = f'Epoch {epoch}'
        losses.append(loss)
        # for k, v in out.items():
        #     if k not in losses:
        #         losses[k] = []
        #     losses[k].append(v.item())
        #     avg_loss = np.mean(losses[k][-50:])
        #     desc += f', {k} {avg_loss:.4f}'

    #     if not quiet:
    #         pbar.set_description(desc)
    #         pbar.update(x.shape[0])
    # if not quiet:
    #     pbar.close()
    return losses


def eval_loss(model, data_loader, quiet):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, eos, y in data_loader:
            pdb.set_trace()
            x = x.cuda()
            eos = eos.cuda()
            y = y.cuda()
            loss = model.loss(x, eos, y)
            total_loss += loss
            # for k, v in out.items():
            #     total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

        # desc = 'Test '
        # for k in total_losses.keys():
        #     total_losses[k] /= len(data_loader.dataset)
        #     desc += f', {k} {total_losses[k]:.4f}'
        # if not quiet:
        #     print(desc)
    return total_loss / len(data_loader.dataset)


def train_epochs(model, train_loader, test_loader, train_args):
    epochs, lr = train_args['epochs'], train_args['lr']
    grad_clip = train_args.get('grad_clip', None)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []
    for epoch in tqdm(range(epochs)):
        train_loss = train(model, train_loader, optimizer, epoch, grad_clip)
        test_loss = eval_loss(model, test_loader)

        train_losses.extend(train_loss)
        test_losses.extend(test_losses)
        # for k in train_loss.keys():
        #     if k not in train_losses:
        #         train_losses[k] = []
        #         test_losses[k] = []
        #     train_losses[k].extend(train_loss[k])
        #     test_losses[k].append(test_loss[k])
        print('epoch: %d, avg_train_loss: %0.2f, avg_test_loss: %0.2f' %\
              (epoch, sum(train_loss)/len(train_loss), test_loss))
    return train_losses, test_losses

def main():
    train_args = {}
    train_args['d_emb'] = 512
    train_args['d_hid'] = 256
    train_args['batch_size'] = 128
    train_args['epochs'] = 5
    train_args['lr'] = 0.001
    train_args['device'] = 'cuda:0'

    train_args['num_class'] = 3 # TODO: connect this to preprocessing

    train_path = "core/dataset/data/processed/train.csv"
    test_path = "core/dataset/data/processed/test.csv"
    tokenizer = word_based()

    train_dataset = twitter_dataset(train_path, tokenizer)
    test_dataset = twitter_dataset(test_path, tokenizer)
    vocab_size = len(tokenizer.id2vocab)
    train_args['vocab_size'] = vocab_size

    train_loader = data.DataLoader(train_dataset, batch_size=train_args['batch_size'], shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=train_args['batch_size'], shuffle=True)

    model = RNNLM(train_args)

    train_epochs(model, train_loader, test_loader, train_args=train_args)
if __name__ == '__main__':
    main()