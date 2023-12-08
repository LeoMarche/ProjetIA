import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.conv2_maxpool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.relu4 = nn.ReLU()
        self.conv4_maxpool = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu6 = nn.ReLU()
        self.conv6_maxpool = nn.MaxPool2d(2)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3)
        self.relu8 = nn.ReLU()
        self.conv8_maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv2_maxpool(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv4_maxpool(x)
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.conv6_maxpool(x)
        x = self.relu7(self.conv7(x))
        x = self.relu8(self.conv8(x))
        x = self.conv8_maxpool(x)
        return torch.sum(x, 1)