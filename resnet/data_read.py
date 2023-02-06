### CIFAR-10부터 시작 -> CIFAR-100으로 VGG-19 및 ResNet 진행
import os
import torch
from torchvision import transforms, datasets

data_dir = './dataset'

train_dataset = datasets.CIFAR10(root=os.path.join(data_dir, 'train'), train=True, download=True, transform=transforms.ToTensor())
tes_dataset = datasets.CIFAR10(root=os.path.join(data_dir, 'test'), train=False, download=True, transform=transforms.ToTensor())

