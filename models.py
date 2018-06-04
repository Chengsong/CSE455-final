import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime
import time

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        if not os.path.exists('logs'):
            os.makedirs('logs')
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S_log.txt')
        self.logFile = open('logs/' + st, 'w')

    def log(self, str):
        print(str)
        self.logFile.write(str + '\n')

    def criterion(self):
        return nn.CrossEntropyLoss()

    def optimizer(self):
        return optim.Adagrad(self.parameters(), lr=0.001)

# Very basic CNN
class CNN(BaseModel):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(18, 6, 3)
        self.fc1 = nn.Linear(6 * 30 * 30, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Implement more advanced model
def common_layer(in_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True)
    )

# ResNeXt
class Resnext(nn.Module):
    # BottleNeck design
    def __init__(self, in_channels, out_channels, stride=1):
        super(Resnext, self).__init__()
        CARDINALITY = 32

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        mid_channels = out_channels / 2

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, stride=1)
        self.common1 = common_layer(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=CARDINALITY)
        self.common2 = common_layer(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, stride=1)
        self.conv3_bn = nn.BatchNorm2d(out_channels)

        # if dimensions for residual doesn't match
        self.conv4 = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        self.conv4_bn = self.conv3_bn

    def forward(self, x):
        res = x

        x = self.conv1(x)
        x = self.common1(x)
        x = self.conv2(x)
        x = self.common2(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)

        if (self.in_channels != self.out_channels or self.stride != 1):
            res = self.conv4(res)
            res = self.conv4_bn(res)

        x += res
        x = F.relu(x)
        return x

class Attention_Module(nn.Module):
    def __init__(self, channels):
        super(Attention_Module, self).__init__()
        # pre-process
        self.res1 = Resnext(channels, channels)

        # trunk branch
        self.trunk = nn.Sequential(
            Resnext(channels, channels),
            Resnext(channels, channels)
        )

        # soft mask branch
        self.softmask = nn.Sequential(
            nn.MaxPool2d
        )


# Model described in Fang Et Al.'s paper about Residual Attention Network with BatchNorm
class Attention_56(BaseModel):
    def __init__(self):
        super(Attention_56, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.res1 = Resnext(64, 256)