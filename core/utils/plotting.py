import sys 
import os
from collections import OrderedDict, Counter
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

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