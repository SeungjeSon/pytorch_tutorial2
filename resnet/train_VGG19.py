""" VGG19 구조 """

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision


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

        self.fc1 = nn.Linear(512*2*2, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 10)

        self.dp = nn.Dropout2d(0.5)



    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1_3(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2_3(x)
        x = self.dp(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.pool3_5(x)
        x = self.dp(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.pool4_5(x)
        x = self.dp(x)

        # print(x.shape)
        x = x.view(-1, 512*2*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.dp(x)
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


## 데이터셋 설정
train_dataset = datasets.CIFAR10(root=os.path.join(data_dir, 'train'), train=True, download=True, transform=transforms.ToTensor())
loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

tes_dataset = datasets.CIFAR10(root=os.path.join(data_dir, 'test'), train=False, download=True, transform=transforms.ToTensor())
loader_test = DataLoader(tes_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ## 데이터셋 확인
# def imshow(img):
#     img = img/2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     print("{0}\t{1}".format(npimg.shape, np.transpose(npimg, (1,2,0)).shape))
#     plt.show()
# dataiter = iter(loader_train)
# images, labels = dataiter.next()
#
# for i, data in enumerate(loader_train, 0):
#     inputs, labels = data
#     labels = labels.to(device)
#     print(labels)
# imshow(torchvision.utils.make_grid(images))


## 네트워크 생성
vgg19 = VGG19().to(device)

## 손실함수 정의
fn_loss = nn.CrossEntropyLoss().to(device)

## optim 설정
optim = torch.optim.Adam(vgg19.parameters(), lr=lr)

## 네트워크 저장
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))

# 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(str.isdigit, f)))

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth'[0]))

    return net, optim, epoch

## 네트워크 학습
st_epoch = 0
# vgg19, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=vgg19, optim=optim)

for epoch in range(num_epoch):
    vgg19.train()
    running_loss = 0.0
    loss_arr = []

    for i, data in enumerate(loader_train, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optim.zero_grad()

        outputs = vgg19(inputs)
        loss = fn_loss(outputs, labels)
        loss.backward()
        optim.step()

        loss_arr += [loss.item()]

        if i % 10 == 0:
            print("Train: EPOCH %04d / %04d | BATCH %04d | Loss %.4f" %
                  (epoch, num_epoch, i, np.mean(loss_arr)))
    if epoch % 5 == 0:
        save(ckpt_dir=ckpt_dir, net=vgg19, optim=optim, epoch=epoch)

