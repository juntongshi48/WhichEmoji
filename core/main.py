import sys 
import os
from collections import OrderedDict, Counter
from tqdm import tqdm
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data

import pdb

sys.path.append(os.getcwd())
# sys.setrecursionlimit(10000)

from core.dataset.preprocessing import preprocessing
from configs.config import Config
from core.dataset.tokenizer import WordBasedTokenizer, PretrainedTokenizer
from core.dataset.twitter_dataset import twitter_dataset
from core.model.RNN import RNNLM, ATTNLM

from core.utils.training import myTrainer
from core.utils.plotting import plot_training_plot, plot_confusion_matrix

# def train(model, train_loader, optimizer, epoch, device, grad_clip=None):
#     model.train()
#     losses = []
#     targets = np.zeros(0)
#     predictions = np.zeros(0)
#     for x, eos, y in train_loader:
#         x = x.to(device)
#         eos = eos.to(device)
#         y = y.to(device)
#         loss = model.loss(x, eos, y)
        
#         optimizer.zero_grad()
#         loss.backward()
#         if grad_clip:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
#         optimizer.step()

#         y_pred = model.predict(x, eos)
#         predictions = np.concatenate((predictions, y_pred))
#         targets = np.concatenate((targets, y.cpu()))

#         # desc = f'Epoch {epoch}'
#         losses.append(loss.cpu().item()) # BUG: transfer loss from cuda to cpu
#         # for k, v in out.items():
#         #     if k not in losses:
#         #         losses[k] = []
#         #     losses[k].append(v.item())
#         #     avg_loss = np.mean(losses[k][-50:])
#         #     desc += f', {k} {avg_loss:.4f}'

#     #     if not quiet:
#     #         pbar.set_description(desc)
#     #         pbar.update(x.shape[0])
#     # if not quiet:
#     #     pbar.close()
#     accuracy = np.sum(predictions == targets) / len(train_loader.dataset)
#     print(f'train_accuracy: {accuracy}')
#     return losses


# def eval(model, data_loader, device):
#     model.eval()
#     total_loss = 0
#     num_batch = 0
#     predictions = np.zeros(0)
#     # targets = np.zeros(0)
#     with torch.no_grad():
#         for x, eos, y in data_loader:
#             x = x.to(device)
#             eos = eos.to(device)
#             y = y.to(device)
#             loss = model.loss(x, eos, y).cpu().item()
#             total_loss += loss
#             num_batch += 1
#             y_pred = model.predict(x, eos)
#             predictions = np.concatenate((predictions, y_pred))
#             # targets = np.concatenate((targets, y.cpu()))
#             # for k, v in out.items():
#             #     total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

#         # desc = 'Test '
#         # for k in total_losses.keys():
#         #     total_losses[k] /= len(data_loader.dataset)
#         #     desc += f', {k} {total_losses[k]:.4f}'
#         # if not quiet:
#         #     print(desc)
#     loss = total_loss / num_batch
#     targets = data_loader.dataset.labels
#     accuracy = np.sum(predictions == targets) / len(data_loader.dataset)
#     f1_macro = f1_score(targets, predictions, average='macro')
#     confus_matrix = confusion_matrix(targets, predictions, normalize='true')
#     eval_metrics = dict(loss=loss, accuracy=accuracy, f1_macro=f1_macro, confusion_matrix=confus_matrix)
#     return eval_metrics

# def evaluate_accuracy(model, data_loader):
#         model.eval()
#         total_correct = 0
#         with torch.no_grad():
#             for x, eos, y in data_loader:
#                 x = x.cuda()
#                 eos = eos.cuda()
#                 y = y.cuda()
#                 y_pred = model.predict(x, eos)
#                 total_correct += torch.sum((y_pred == y))

#         return total_correct / len(data_loader.dataset)


# def train_epochs(model, train_loader, val_loader, test_loader, cfg):
#     epochs, lr = cfg.epochs, cfg.lr
#     grad_clip = cfg.grad_clip if hasattr(cfg, "grad_clip") else None
    
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     device = cfg.device

#     train_losses, val_losses, val_accuracies, test_metrics_ = [], [], [], []
#     for epoch in tqdm(range(epochs)):
#         model.epoch = epoch
#         train_loss = train(model, train_loader, optimizer, epoch, device, grad_clip)
#         val_metrics = eval(model, val_loader, device)
#         test_metrics = eval(model, test_loader, device)

#         train_losses.extend(train_loss)
#         val_losses.append(val_metrics['loss'])
#         val_accuracies.append(val_metrics['accuracy'])
#         test_metrics_.append(test_metrics)
#         print('epoch: %d, avg_train_loss: %0.8f, avg_val_loss: %0.8f, val_accuracy: %0.4f' %\
#               (epoch, sum(train_loss)/len(train_loss), val_metrics['loss'], val_metrics['accuracy']))
        
#     return train_losses, val_losses, val_accuracies, test_metrics_

def main(cfg):
    id = list(range(len(cfg.data.labels)))
    id2label = dict(zip(id, cfg.data.labels))

    preprop = preprocessing(id2label, min_sentence_len=10, multi_class_label=(cfg.num_output_labels>1))
    preprop.process_all_csvs_in_directory()

    ## dataset and tokenizer
    train_path = os.path.join(cfg.data.data_dir, "train.csv")
    val_path = os.path.join(cfg.data.data_dir, "val.csv")
    test_path = os.path.join(cfg.data.data_dir, "test.csv")
    
    tokenizer = None
    kwargs = dict(min_sentence_len=cfg.data.min_sentence_len, unk_max_frequency=cfg.data.unk_max_frequency)
    if cfg.data.tokenizer == "word_based":
        tokenizer = WordBasedTokenizer(**kwargs)
    elif cfg.data.tokenizer == "pretrained":
        if hasattr(cfg.data, "pretrained_name"):
            tokenizer = PretrainedTokenizer(cfg.data.pretrained_name, **kwargs)
        else:
            tokenizer = PretrainedTokenizer(**kwargs) # bert tokenizer is used by default
    else:
        raise NotImplementedError(f"Unknown tokenizer {cfg.data.tokenizer}")

    train_dataset = twitter_dataset(train_path, tokenizer, train=True)
    val_dataset = twitter_dataset(val_path, tokenizer)
    test_dataset = twitter_dataset(test_path, tokenizer)

    ## dataloader
    train_loader = data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    ## Print data split stats
    loaders = [train_loader, val_loader, test_loader]

    for loader in loaders:
        labels = loader.dataset.labels
        u, counts = np.unique(labels, return_counts=True)
        print(u)
        print(counts)
        print()
    ## -------------------------------------

    ## Select a model
    num_class = len(id)
    vocab_size = len(tokenizer.vocab2id)
    print(f"Vocab Size: {vocab_size}")
    if cfg.model.attention:
        model = ATTNLM(cfg, num_class, vocab_size).to(cfg.device)
    else:
        model = RNNLM(cfg, num_class, vocab_size).to(cfg.device)

    ## Training
    trainer = myTrainer(model, train_loader, val_loader, test_loader, cfg)
    train_losses, val_losses, val_accuracies, test_metrics = trainer.train_epochs()
    
    # Draw Training Plot
    plot_training_plot(train_losses, val_losses, 'Training_Plot', 'training_plot.png')
    
    # Select Best epoch
    criterion = val_accuracies
    best_epoch = np.argmax(np.array(criterion))
    print(f'\nThe best epoch is {best_epoch}, with accuracy: {val_accuracies[best_epoch]} ')
    test_metrics_best = test_metrics[best_epoch]

    ## Summarize Testing Metrics
    print(f'\n TESTING METRICS')
    print(f'Test Accuracy: {test_metrics_best["accuracy"]}')
    print(f'Test Macro F1: {test_metrics_best["f1_macro"]}')

    # Draw Confusion Matrix
    att = "att" if cfg.model.attention else "noatt"
    # /core/results/confusion_matrix/
    filename = f'confusion_matrix/CM\
_demb{cfg.model.d_embedding}\
_dhid{cfg.model.d_hidden}\
_nlay{cfg.model.n_layer}\
_bs{cfg.batch_size}\
_{att}\
_{cfg.data.tokenizer}\
.png'
    if cfg.num_output_labels == 1:
        plot_confusion_matrix(id2label.values(), test_metrics_best['confusion_matrix'], 'Confusion_Matrix', filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file",
                        type=str,)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--num_output_labels", type=int, default=1)
    _args = parser.parse_args()
    cfg = Config(**_args.__dict__)
    print(f"The config of this experiment is : \n {cfg}")
    main(cfg)