import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.utils import make_grid

import timm

from tqdm import tqdm
import PIL.Image as Image

# import some common libraries
import numpy as np
import os, json, cv2, random

import matplotlib.pyplot as plt


batch_size = 32
epochs = 30


# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(0.4), 
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

data_val_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

data_test_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('croped_dataset/train_images', transform=data_transforms)
val_dataset = datasets.ImageFolder('croped_dataset/val_images', transform=data_val_transforms)
test_dataset = datasets.ImageFolder('croped_dataset/test_images',transform=data_test_transforms)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size, shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size, shuffle=True, num_workers=1)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1, shuffle=False, num_workers=1)