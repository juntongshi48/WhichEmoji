import sys 
import os
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

def multi_center_gaussian(N, *sample_shape, centers: list=None, energy: list=None):
    # Returns a tensor of N samples drawn from a specified mixture of gaussians
        if centers is None:
            centers = [np.zeros(sample_shape)] # one center by default
            # assert energy is None # energy cannot be not None if centers is None
        if energy is None:
            energy = np.ones((len(centers))) # equi-energy by default
        assert len(energy) == len(centers) , "centers and energy must have the same length"
        centers = np.array(centers, dtype=float) # convert centers into a nparr
        assert all(center.shape == sample_shape for center in centers) , "center in centers must have the same shape as sample_shape"

        energy = torch.tensor(energy, dtype=torch.float)
        z = torch.multinomial(energy, N, replacement=True) # Bug: z = torch.multinomial(energy, N, replacement=True)--replacement is clear to false by default, thus whenever we want to sample a lot samples from a few values, we need to explictily set replacement
        displacements = torch.tensor(centers, dtype=torch.float)[z] # N x sample_shape
        samples = torch.randn(N, *sample_shape, dtype=torch.float) + displacements
        return samples

class Mixture_of_two_2DGaussians (data.Dataset):
    def __init__(self, N: int =1000, center_1: tuple =(5,5), center_2: tuple =(-5,-5)):
        self.data = multi_center_gaussian(N, 2, centers=[center_1, center_2])
        self.data = self.data.numpy()
        self.data = np.float32(self.data)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]