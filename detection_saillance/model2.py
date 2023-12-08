import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv2_maxpool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv4_maxpool = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3)
        self.conv6_maxpool = nn.MaxPool2d(2)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3)
        self.conv8_maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv2_maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv4_maxpool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv6_maxpool(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv8_maxpool(x)
        return torch.sum(x, 1)