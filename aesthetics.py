import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial
from torchvision.io import read_image
from torchvision.transforms import Resize
from torchvision.io import ImageReadMode
import argparse

## Returns either the cpu if nod GPU is available or the first CUDA capable device else
def get_optimal_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

## Loads and return a Net with the given weights parametrized
def load_premade_model(weight_path, device):
    model = Net()
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    model.to(device)
    return model

## Run inference on possibly multiple images
def inference_torch(model, torch_images, device):
    r = Resize((256, 256), antialias=True)
    if device.type == 'cuda':
        inp = torch.div(r.forward(torch_images), 255.0).type(torch.cuda.FloatTensor)
    else:
        inp = torch.div(r.forward(torch_images), 255.0).type(torch.FloatTensor)
    inp.to(device)
    print(inp.shape)
    with torch.no_grad():
        res = model(inp)
    print(res.shape)
    res = res.cpu()
    return res.squeeze().detach().numpy()

def inference(model, image_path, device):
    img = read_image(image_path, ImageReadMode.RGB).unsqueeze(0)
    print(img.shape)
    return inference_torch(model, img, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights")
    parser.add_argument("image")
    args = parser.parse_args()
    device = get_optimal_device()
    model = load_premade_model(args.weights, device)
    print(inference(model, args.image, device))