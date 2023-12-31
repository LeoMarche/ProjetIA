import torch
from torchvision.io import read_image
from torchvision.transforms import Resize
from torchvision.io import ImageReadMode
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial
import argparse
import matplotlib.pyplot as plt
import points_interets
import recadrage
import aesthetics
import numpy as np
import cv2

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_det_weights")
    parser.add_argument("aesthetics_weights")
    parser.add_argument("image")
    parser.add_argument("ratio")
    args = parser.parse_args()

    device = get_optimal_device()
    model = load_premade_model(args.feature_det_weights, device)
    r, initial_shape = inference(model, args.image, device)
    r = r * (255/np.max(r))
    r = (r > 50) * r
    
    plt.imshow(r, cmap='gray', vmin=0, vmax=255)
    plt.title("saliency result")
    plt.show()

    interest_clusters = points_interets.interest_clusters(r)

    potential_crops = []

    for i in range(len(interest_clusters)):
        crop_tuple = recadrage.get_crop_tuple_one_center(float(args.ratio), r, initial_shape, interest_clusters[i])
        potential_crops.append(recadrage.crop_image(args.image, crop_tuple))
    crop_tuple = recadrage.get_crop_tuple_using_least_square_distance_to_interest_points(float(args.ratio), r.shape, initial_shape, [i['centroid'] for i in interest_clusters])
    potential_crops.append(recadrage.crop_image(args.image, crop_tuple))

    crop_tuple = recadrage.get_crop_tuple_using_1D_saliency(float(args.ratio), r, initial_shape)
    potential_crops.append(recadrage.crop_image(args.image, crop_tuple))

    resi = Resize((256, 256), antialias=True)

    for i in range(len(potential_crops)):
        tmp = cv2.cvtColor(potential_crops[i], cv2.COLOR_BGR2RGB)
        tens = resi.forward(torch.mul(transforms.ToTensor()(tmp), 255.0).type(torch.IntTensor).unsqueeze(0))
        potential_crops[i] = tens
    
    inp = torch.Tensor(len(potential_crops), 3, 256, 256)
    torch.cat(potential_crops, out=inp)
    print(inp[0].shape, potential_crops[0].shape)

    aes_model = aesthetics.load_premade_model(args.aesthetics_weights, device)
    res = aesthetics.inference_torch(aes_model, inp, device)

    print(res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()