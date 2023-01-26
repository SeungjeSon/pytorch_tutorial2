""" VGG19 구조 """

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


## 트레이닝 파라미터 설정
lr = 1e-3
batch_size = 64
num_epoch = 10

data_dir = './dataset'
ckpt_dir = './checkpoint'
log_dir = './log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

## 네트워크 구축
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        def CBR2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True):
            layer = []
            layer += [nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)]
            layer += [nn.BatchNorm2d(num_features=out_channel)]
            layer += [nn.ReLU()]

            cbr = nn.Sequential(*layer)

            return cbr
        self.conv1_1 = CBR2d(in_channel=3, out_channel=64)
        self.conv1_2 = CBR2d(in_channel=64, out_channel=64)
        self.pool1_3 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = CBR2d(in_channel=64, out_channel=128)
        self.conv2_2 = CBR2d(in_channel=128, out_channel=128)
        self.pool2_3 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = CBR2d(in_channel=128, out_channel=256)
        self.conv3_2 = CBR2d(in_channel=256, out_channel=256)
        self.conv3_3 = CBR2d(in_channel=256, out_channel=256)
        self.conv3_4 = CBR2d(in_channel=256, out_channel=256)
        self.pool3_5 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = CBR2d(in_channel=256, out_channel=512)
        self.conv4_2 = CBR2d(in_channel=512, out_channel=512)
        self.conv4_3 = CBR2d(in_channel=512, out_channel=512)
        self.conv4_4 = CBR2d(in_channel=512, out_channel=512)
        self.pool4_5 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(256, 64)
        self.fc7 = nn.Linear(64, 10)

        self.dp = nn.Dropout2d(0.5)



    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)
        pool1_3 = self.pool1_3(conv1_2)

        conv2_1 = self.conv2_1(pool1_3)
        conv2_2 = self.conv2_2(conv2_1)
        pool2_3 = self.pool2_3(conv2_2)
        dp1 = self.dp(pool2_3)

        conv3_1 = self.conv3_1(dp1)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        conv3_4 = self.conv3_4(conv3_3)
        pool3_5 = self.pool3_5(conv3_4)
        dp2 = self.dp(pool3_5)

        conv4_1 = self.conv4_1(dp2)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(conv4_2)
        conv4_4 = self.conv4_4(conv4_3)
        pool4_5 = self.pool4_5(conv4_4)
        dp3 = self.dp(pool4_5)

        fc0 = nn.Linear(torch.flatten(dp3, 1), 4096)
        fc1 = self.fc1(fc0)
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)
        fc4 = self.fc4(fc3)
        dp4 = self.dp(fc4)
        fc5 = self.fc5(dp4)
        fc6 = self.fc6(fc5)
        x = self.fc7(fc6)

        return x


## 네트워크 학습
train_dataset = datasets.CIFAR10(root=os.path.join(data_dir, 'train'), train=True, download=True, transform=transforms.ToTensor())
loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

tes_dataset = datasets.CIFAR10(root=os.path.join(data_dir, 'test'), train=False, download=True, transform=transforms.ToTensor())
loader_test = DataLoader(tes_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


## 네트워크 생성
vgg19 = VGG19().to(device)

## 손실함수 정의
fn_loss = nn.CrossEntropyLoss().to(device)

## optim 설정
optim = torch.optim.Adam(vgg19.parameters(), lr=lr)

