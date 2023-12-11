import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fine_features=models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[0:30]
        self.coarse_features=models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[0:30]

        self.upsample=partial(F.interpolate, mode='bilinear', antialias=True)
        self.integrate=nn.Conv2d(1024,1,(1,1))
        self.sigm=nn.Sigmoid()

    def forward(self, x):
        coarse_x = nn.functional.interpolate(x, 240)
        fine_saliency = self.fine_features(x)
        coarse_saliency = self.coarse_features(coarse_x)
        H, W = fine_saliency.shape[-2], fine_saliency.shape[-1]

        coarse_saliency = self.upsample(coarse_saliency, size = (H, W))

        out = torch.cat([fine_saliency, coarse_saliency], 1)
        out = self.integrate(out)
        out = self.sigm(out)

        return out