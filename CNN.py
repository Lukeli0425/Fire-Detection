import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3), # 6*48*48
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2) # 6*24*24
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 32, 5), # 12*20*20
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # 12*10*10
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3), # 64*8*8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # 32*4*4
        )
        
        self.fc1 = nn.Linear(512, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 3)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y  = self.dropout(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu4(y)
        y = self.fc2(y)
        y = self.relu5(y)
        # y = self.fc3(y)
        # y = self.relu6(y)
        y = self.softmax(y)
        return y