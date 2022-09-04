#a convolutional neural net to recognize digits from mnist set - jasper hilliard
import torch
import torchvision
from torch import nn
from torch import optim
from torch import tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt

#hyperparameters
batch_size = 64

#data loader
training_data = torchvision.datasets.mnist('mnist_data', train=True, download=True)
val_dataset =  torchvision.datasets.mnist('mnist_data', train=False, download=True)

#transform to tensor