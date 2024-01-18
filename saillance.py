from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial

## Define the Neural Network architecture (import from detection_saillance/premade.py)
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fine_features=models.vgg16().features[0:30]
        self.coarse_features=models.vgg16().features[0:30]

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

## Returns either the cpu if nod GPU is available or the first CUDA capable device else
def get_optimal_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Loads and return a Net with the given weights parametrized
def load_premade_model(weight_path, device):
    model  = Net()
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    model.to(device)
    return model

## Compute the saillance detection on a signe image
def inference(model, image_path, device):
    if device.type == 'cuda':
        data = torch.div(read_image(image_path, ImageReadMode.RGB).type(torch.cuda.FloatTensor), 255.0)
    else:
        data = torch.div(read_image(image_path, ImageReadMode.RGB).type(torch.FloatTensor), 255.0)
    initial_shape = (data.shape[-1], data.shape[-2])
    r = Resize(480, antialias=True)
    img = r.forward(data).unsqueeze(0)
    img.to(device)
    res = model(img)
    tr_res = torch.mul(res, 255.0).cpu()
    return tr_res.squeeze().detach().numpy(), initial_shape
