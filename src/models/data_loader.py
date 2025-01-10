"""
data_loader.py

Provides the `ImageData` class for loading images and features from a Pandas 
DataFrame. The DataFrame must include image file paths and additional features 
for model training or evaluation.

Classes:
    ImageData: Loads and processes images and features for PyTorch models.
"""

import pandas as pd
from PIL import Image
import numpy as np

import torch

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

# Currently not transforming Images!!

transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load Data

dataset = pd.read_csv('data/spectrograms/numerical_labels.csv')

#numerical_data = dataset.drop("filepath", axis=1)

# Split data first into training and testing so that we preserve the target variable (diagnosis) in both
training_set, testing_set = train_test_split(dataset, test_size=0.15, shuffle=True)

# filepaths
training_paths = training_set["filepath"]
testing_paths = testing_set["filepath"]

# Numerical feature datasets
training_features = training_set.drop(["filepath", "diagnosis"], axis=1)
testing_features = testing_set.drop(["filepath", "diagnosis"], axis=1)

# Numerical target datasets - specify that they're dataframes otherwise it will be a Series
training_target = pd.DataFrame(training_set["diagnosis"])
testing_target = pd.DataFrame(testing_set["diagnosis"])
# Make a np array of training images

img_list = []
for path in training_paths:
    img = Image.open(path)
    img_array = np.array(img)
    img_list.append(img_array)

image_array = np.array(img_list)

# Normalize Data

scaler = MinMaxScaler()
n_training_features = pd.DataFrame(scaler.fit_transform(training_features))
n_testing_features = pd.DataFrame(scaler.fit_transform(testing_features))
n_training_target = pd.DataFrame(scaler.fit_transform(training_target))
n_testing_target = pd.DataFrame(scaler.fit_transform(testing_target))

# Creating Tensors
t_train_features = torch.Tensor(n_training_features.to_numpy())
t_train_targets = torch.Tensor(n_training_target.to_numpy())
t_train_images = torch.Tensor(image_array)

# Training dataloader
combined_tensor = TensorDataset(t_train_images, t_train_features, t_train_targets)
train_dataloader = DataLoader(combined_tensor, batch_size=32, shuffle=True)