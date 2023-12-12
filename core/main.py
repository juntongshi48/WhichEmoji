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

def main(cfg):
    ## Check GPU availability
    if torch.cuda.is_available():
        cfg.device = 'cuda'

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
    train_losses, val_losses, train_accuracies, val_accuracies, test_metrics = trainer.train_epochs()
    
    # Draw Training Plot
    if not os.path.exists("training_plot"):
        os.mkdir("training_plot")
    title = f"training_plot_class_{cfg.num_output_labels}"
    if cfg.num_output_labels > 1:
        title = f"training_plot_class_{cfg.num_output_labels}_{cfg.model.threshold}_{cfg.model.pos_weight}"
    plot_training_plot(train_losses, val_losses, train_accuracies, val_accuracies, title, f'training_plot/{title}.png')
    
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
        if not os.path.exists("confusion_matrix"):
            os.mkdir("confusion_matrix")
        plot_confusion_matrix(id2label.values(), test_metrics_best['confusion_matrix'], 'Confusion_Matrix', filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file",
                        type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--num_output_labels", type=int, default=1)
    _args = parser.parse_args()
    cfg = Config(**_args.__dict__)
    print(f"The config of this experiment is : \n {cfg}")
    main(cfg)