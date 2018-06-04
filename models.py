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
