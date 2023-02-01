import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

# parameters
lr = 1e-3
num_epoch = 10
batch_size = 4

# path
data_dir = './dataset'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#데이터 불러오기, 학습여부  o
trainset = torchvision.datasets.CIFAR10(root=os.path.join(data_dir, 'train'), train=True,
                                        download=True, transform=transform)

#학습용 셋은 섞어서 뽑기
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
#데이터 불러오기, 학습여부  x
testset = torchvision.datasets.CIFAR10(root=os.path.join(data_dir, 'test'), train=False,
                                       download=True, transform=transform)
#테스트 셋은 굳이 섞을 필요가 없음
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
#클래스들
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# #이미지 확인하기
#
# def imshow(img):
#     img = img / 2 + 0.5     # 정규화 해제
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# # 학습용 이미지 뽑기
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# # 이미지 보여주기
# imshow(torchvision.utils.make_grid(images))
#
# # 이미지별 라벨 (클래스) 보여주기
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

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

net = VGG19()

fn_loss = nn.CrossEntropyLoss().to(device)
optim = optim.Adam(net.parameters(), lr=lr)

# 네트워크 학습
for epoch in range(num_epoch):
    running_rate = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optim.zero_grad()
        output = net(inputs)
        loss = fn_loss(output, labels)
        loss.backward()
        optim.step()

        # 결과 출력
        running_rate += loss.item()
        if i % 2000 == 1999: # 2000개마다
            print('[%d, %5d] loss: %.3f'%(epoch, i, running_rate/2000))
            running_rate = 0.0

print('Finish Training')

# 학습한 모델 저장
torch.save(net.state_dict(), os.path.join(data_dir, 'model'))












