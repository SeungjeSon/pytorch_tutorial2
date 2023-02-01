import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# func
def imshow(img):
    img = img / 2 + 0.5     # 정규화 해제
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #input = 3, output = 6, kernal = 5
        self.conv1 = nn.Conv2d(3, 6, 5)
        #kernal = 2, stride = 2, padding = 0 (default)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #input feature, output feature
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 값 계산
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x





# paramter set
batch_size = 10

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#데이터 불러오기, 학습여부  x
testset = torchvision.datasets.CIFAR10(root='./dataset/test', train=False,
                                       download=True, transform=transform)
#테스트 셋은 굳이 섞을 필요가 없음
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

#클래스들
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(testloader)
images, labels = dataiter.next()
images = images.to(device)
labels = labels.to(device)


# 학습한 모델로 예측값 출력
PATH = './dataset/model/net.pth'
net = Net().to(device)
# net = VGG19().to(device)
net.load_state_dict(torch.load(PATH))
outputs = net(images)

print('GroundTruth: ', ' '.join('%5s'%classes[labels[j]] for j in range(10)))
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s'% classes[predicted[j]] for j in range(10)))
# 실험용 데이터와 결과 출력
imshow(torchvision.utils.make_grid(images))
