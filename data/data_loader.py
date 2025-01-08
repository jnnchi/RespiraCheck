"""
dataloader.py

Provides the `ImageData` class for loading images and features from a Pandas 
DataFrame. The DataFrame must include image file paths and additional features 
for model training or evaluation.

Classes:
    ImageData: Loads and processes images and features for PyTorch models.
"""

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = pd.read_csv('data/spectrograms/labels.csv')


# class ImageData(Dataset):
#     def __init__(self, dataframe, transform=None):
#         self.dataframe = dataframe
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataframe)
    
#     def __getitem__(self, idx):
#         img_path = self.dataframe.iloc[idx, 0]
#         label = self.dataframe.iloc[idx, 1]

#         image = Image.open(img_path).convert('RGB')

#         if self.transform:
#             image = self.transform(image)


# dataset_object = ImageData(dataframe=dataset, transform=transform)

# print(dataset_object[1])

 