import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import os, shutil
import numpy as np


""" 학습 파라미터 설정 """
lr = 1e-4
batch_size = 20
num_epoch = 15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


""" Path """
origin_data_dir = './dataset/kaggle/train'
base_dir = './dataset/dataset_small_kaggle'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

log_dir = os.path.join(base_dir, 'log')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
val_cats_dir = os.path.join(val_dir, 'cats')
val_dogs_dir = os.path.join(val_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

if not os.path.exists(train_cats_dir):
    os.makedirs(train_cats_dir)
if not os.path.exists(train_dogs_dir):
    os.makedirs(train_dogs_dir)
if not os.path.exists(val_cats_dir):
    os.makedirs(val_cats_dir)
if not os.path.exists(val_dogs_dir):
    os.makedirs(val_dogs_dir)
if not os.path.exists(test_cats_dir):
    os.makedirs(test_cats_dir)
if not os.path.exists(test_dogs_dir):
    os.makedirs(test_dogs_dir)


""" 네트워크 구축 """
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # def CBR2d(in_channel, out_channel, kerner_size=3, stride=1, padding=1, bias=True):
        #     layer = []
        #     layer += [nn.Conv2d(in_channel=in_channel, out_channel=out_channel, kerner_size=kerner_size, stride=stride, padding=padding, bias=bias)]
        #     layer += [nn.BatchNorm2d(num_features=out_channel)]
        #     layer += [nn.ReLU()]
        #
        #     cbr = nn.Sequential(*layer)
        #
        #     return cbr

        # self.conv1 = CBR2d(in_channel=3, out_channel=32)
        # self.conv2 = CBR2d(in_channel=32, out_channel=64)
        # self.conv3 = CBR2d(in_channel=64, out_channel=128)
        # self.conv4 = CBR2d(in_channel=128, out_channel=128)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        # self.bn

        self.fc1 = nn.Linear(128*9*9, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        # print(x.shape)
        x = x.view(-1, 128*9*9)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

transform = transforms.Compose([transforms.Resize((150, 150)),
                                transforms.ToTensor()])
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)



## 학습한 모델 Load
PATH = './dataset/dataset_small_kaggle/model/net.pth'
net = Net().to(device)
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
net.eval()
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = net(inputs)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 500 test images: %d %%' % (100 * correct / total))
