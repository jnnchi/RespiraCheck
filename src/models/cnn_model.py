"""
cnn_model.py: Defines a Convolutional Neural Network for mel-spectrogram classification.

This module includes the model architecture and any necessary helper functions for 
defining and modifying the CNN.

Classes:
    CNNModel: A PyTorch implementation of a basic CNN for image classification.
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv

import torchvision.transforms as transforms

# Using a 70% training, 15% validation, and 15% testing data split