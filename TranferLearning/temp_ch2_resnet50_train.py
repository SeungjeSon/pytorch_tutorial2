import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets, models
from torchsummary import summary

import matplotlib.pyplot as plt
import os, shutil
import numpy as np


""" 학습 파라미터 설정 """
lr = 1e-3
batch_size = 50
num_epoch = 30

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


""" datasets 분활 """
# cats
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(origin_data_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(origin_data_dir, fname)
    dst = os.path.join(val_cats_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(origin_data_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

# dogs
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(origin_data_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(origin_data_dir, fname)
    dst = os.path.join(val_dogs_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(origin_data_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

# 분활된 데이터수 확인
print("고양이 데이터셋")
print("tran : {0}\t|\tval : {1}\t|\ttest : {2}".format(len(os.listdir(train_cats_dir)), len(os.listdir(val_cats_dir)), len(os.listdir(test_cats_dir))))
print()
print("강아지 데이터셋")
print("tran : {0}\t|\tval : {1}\t|\ttest : {2}".format(len(os.listdir(train_dogs_dir)), len(os.listdir(val_dogs_dir)), len(os.listdir(test_dogs_dir))))


""" 네트워크 구축 """
resnet50_pretrained = models.resnet50(pretrained=True)
# print(resnet50_pretrained)
num_classes = 2
num_ftrs = resnet50_pretrained.fc.in_features
resnet50_pretrained.fc = nn.Linear(num_ftrs, num_classes)

net = resnet50_pretrained.to(device)
summary(resnet50_pretrained, input_size=(3, 150, 150), device=device.type)

transform = transforms.Compose([transforms.Resize((150, 150)),
                                transforms.RandomHorizontalFlip(p=0.2),
                                transforms.RandomVerticalFlip(p=0.2),
                                # transforms.RandomAffine((-90, 90), translate=(0.2, 0.2)),
                                transforms.RandomRotation(45),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# net = resnet50_pretrained().to(device)
fn_loss = nn.CrossEntropyLoss().to(device)
optim = optim.Adam(net.parameters(), lr=lr)
writer = SummaryWriter(log_dir=log_dir)

for epoch in range(num_epoch):
    net.train()
    runnung_rate = 0.0
    loss_arr = []
    acc_arr = []

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        # labels = labels.unsqueeze(1)      # BCELoss 일떄 사용
        labels = labels.to(device)


        optim.zero_grad()
        output = net(inputs)
        # loss = fn_loss(output.to(torch.float32), labels.to(torch.float32))        # BCELoss 일떄 사용

        loss = fn_loss(output, labels)
        loss.backward()
        optim.step()


        runnung_rate += loss.item()
        loss_arr += [loss.item()]
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch, i, runnung_rate/10))
            runnung_rate = 0.0

    writer.add_scalar('loss', np.mean(loss_arr), epoch)

    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = net(inputs)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


        print('Accuracy of the network on the 500 test images: %d %%' % (100 * correct / total))
    writer.add_scalar('acc', 100 * correct / total, epoch)



print("Finish Training")

# 학습한 모델 저장
PATH = './dataset/dataset_small_kaggle/model/net_resnet50.pth'
save_dir = './dataset/dataset_small_kaggle/model'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
torch.save(net.state_dict(), PATH)

writer.close()
