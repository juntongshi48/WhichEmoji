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

from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

import pdb

from tokenizer import word_based
from twitter_dataset import twitter_dataset
from RNN import RNNLM, ATTNLM


def plot_training_plot(train_losses, val_losses, title, fname):
    plt.figure()
    n_epochs = len(val_losses)-1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_val = np.arange(n_epochs+1)

    plt.plot(x_train, train_losses, label='CE_train')
    plt.plot(x_val, val_losses, label='CE_val')

    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(fname)

def plot_confusion_matrix(labels, confusion_matrix, title, fname):
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix)

    # Show ticks and label them
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Rotate x labels and and set label alignments
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")

    # Label each cell by their corresponding intensty 
    confusion_matrix = np.round(confusion_matrix, decimals=2)
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, confusion_matrix[i,j], ha="center", va="center", color="w", fontsize='x-small')
    
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(fname)
    
    # ## Reference: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    # ax.spines[:].set_visible(False)
    # ax.grid(which="major", color="w", linestyle='-', linewidth=3)
    # ax.tick_params(which="major", bottom=True, left=True)



def train(model, train_loader, optimizer, epoch, grad_clip=None, rectify=False):
    model.train()
    losses = []
    targets = np.zeros(0)
    predictions = np.zeros(0)
    for x, eos, y in train_loader:
        x = x.cuda()
        eos = eos.cuda()
        y = y.cuda()
        loss = model.loss(x, eos, y)
        
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        y_pred = model.predict(x, eos)
        predictions = np.concatenate((predictions, y_pred))
        targets = np.concatenate((targets, y.cpu()))

        # desc = f'Epoch {epoch}'
        losses.append(loss.cpu().item()) # BUG: transfer loss from cuda to cpu
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
    accuracy = np.sum(predictions == targets) / len(train_loader.dataset)
    print(f'train_accuracy: {accuracy}')
    return losses


def eval(model, data_loader):
    model.eval()
    total_loss = 0
    num_batch = 0
    predictions = np.zeros(0)
    # targets = np.zeros(0)
    with torch.no_grad():
        for x, eos, y in data_loader:
            x = x.cuda()
            eos = eos.cuda()
            y = y.cuda()
            loss = model.loss(x, eos, y).cpu().item()
            total_loss += loss
            num_batch += 1
            y_pred = model.predict(x, eos)
            predictions = np.concatenate((predictions, y_pred))
            # targets = np.concatenate((targets, y.cpu()))
            # for k, v in out.items():
            #     total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

        # desc = 'Test '
        # for k in total_losses.keys():
        #     total_losses[k] /= len(data_loader.dataset)
        #     desc += f', {k} {total_losses[k]:.4f}'
        # if not quiet:
        #     print(desc)
    loss = total_loss / num_batch
    targets = data_loader.dataset.labels
    accuracy = np.sum(predictions == targets) / len(data_loader.dataset)
    f1_macro = f1_score(targets, predictions, average='macro')
    confus_matrix = confusion_matrix(targets, predictions, normalize='true')
    eval_metrics = dict(loss=loss, accuracy=accuracy, f1_macro=f1_macro, confusion_matrix=confus_matrix)
    return eval_metrics

def evaluate_accuracy(model, data_loader):
        model.eval()
        total_correct = 0
        with torch.no_grad():
            for x, eos, y in data_loader:
                x = x.cuda()
                eos = eos.cuda()
                y = y.cuda()
                y_pred = model.predict(x, eos)
                total_correct += torch.sum((y_pred == y))

        return total_correct / len(data_loader.dataset)


def train_epochs(model, train_loader, val_loader, test_loader, train_args):
    epochs, lr = train_args['epochs'], train_args['lr']
    grad_clip = train_args.get('grad_clip', None)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, val_accuracies, test_metrics_ = [], [], [], []
    for epoch in tqdm(range(epochs)):
        model.epoch = epoch
        train_loss = train(model, train_loader, optimizer, epoch, grad_clip)
        val_metrics = eval(model, val_loader)
        test_metrics = eval(model, test_loader)

        train_losses.extend(train_loss)
        val_losses.append(val_metrics['loss'])
        val_accuracies.append(val_metrics['accuracy'])
        test_metrics_.append(test_metrics)
        print('epoch: %d, avg_train_loss: %0.8f, avg_val_loss: %0.8f, val_accuracy: %0.4f' %\
              (epoch, sum(train_loss)/len(train_loss), val_metrics['loss'], val_metrics['accuracy']))
        
    return train_losses, val_losses, val_accuracies, test_metrics_

def main():
    id2label = {0: "enraged_face", 
             1: "face_holding_back_tears", 
             2:"face_savoring_food", 
             3: "face_with_tears_of_joy", 
             4: "fearful_face", 
             5: "hot_face", 
             6: "sun", 
             7: "loudly_crying_face", 
             8: "smiling_face_with_sunglasses", 
             9: "thinking_face"}
    train_args = {}
    train_args['d_emb'] = 512
    train_args['d_hid'] = 128
    train_args['n_layer'] = 5
    train_args['batch_size'] = 256
    train_args['epochs'] = 80
    train_args['lr'] = 5e-5
    train_args['device'] = 'cuda:0'
    train_args['attention'] = True

    # train_args['d_emb'] = 512
    # train_args['d_hid'] = 64
    # train_args['n_layer'] = 1
    # train_args['batch_size'] = 128
    # train_args['epochs'] = 2
    # train_args['lr'] = 5e-4
    # train_args['device'] = 'cuda:0'

    train_args['num_class'] = len(id2label) # TODO: connect this to preprocessing

    train_path = "core/dataset/data/processed/train.csv"
    val_path = "core/dataset/data/processed/val.csv"
    test_path = "core/dataset/data/processed/test.csv"
    tokenizer = word_based()

    train_dataset = twitter_dataset(train_path, tokenizer)
    val_dataset = twitter_dataset(val_path, tokenizer)
    test_dataset = twitter_dataset(test_path, tokenizer)
    vocab_size = len(tokenizer.id2vocab)
    train_args['vocab_size'] = vocab_size

    train_loader = data.DataLoader(train_dataset, batch_size=train_args['batch_size'], shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=train_args['batch_size'], shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=train_args['batch_size'], shuffle=False)

    ## Select a model
    if train_args['attention']:
        model = ATTNLM(train_args).to(train_args['device'])
    else:
        model = RNNLM(train_args).to(train_args['device'])


    ## Training
    train_losses, val_losses, val_accuracies, test_metrics = train_epochs(model, train_loader, val_loader, test_loader, train_args=train_args)
    
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
    att = "att" if train_args["attention"] else "noatt"
    # /core/results/confusion_matrix/
    filename = f'CM_{train_args["d_emb"]}_{train_args["d_hid"]}_{train_args["n_layer"]}_{train_args["batch_size"]}_{att}.png'
    plot_confusion_matrix(id2label.values(), test_metrics_best['confusion_matrix'], 'Confusion_Matrix', filename)


if __name__ == '__main__':
    main()