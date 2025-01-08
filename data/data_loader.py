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

# Split Data

numerical_data = dataset.drop("filepath", axis=1)
labels = dataset["diagnosis"]
features = dataset.drop("diagnosis", axis=1)

training_features, testing_features, training_target, testing_target = train_test_split(features, labels.to_frame(), test_size=0.15, shuffle=False)

training_paths = training_features["filepath"]
training_features.drop("filepath")

testing_paths = testing_features["filepath"]
testing_features.drop("filepath")

print(training_features)
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

combined_tensor = TensorDataset(t_train_images, t_train_features, t_train_targets)

train_dataloader = DataLoader(combined_tensor, batch_size=32, shuffle=True)


 # Dataloader