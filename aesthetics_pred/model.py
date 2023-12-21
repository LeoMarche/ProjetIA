import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.features_det=models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.feature_eval=nn.Linear(in_features=1000, out_features=1)
    
    def forward(self, x):
        x = self.features_det(x)
        x = self.lrelu(x)
        return self.feature_eval(x)

    
