import torch
import premade as mod
from torchvision.io import read_image
from torchvision.transforms import Resize
from torchvision.io import ImageReadMode
import matplotlib.pyplot as plt

model  = mod.Net()
model.load_state_dict(torch.load("./results/premade.pth"))
model.eval()

path = "./dataset/data/images/val/COCO_val2014_000000001700.jpg"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    data = torch.div(read_image(path, ImageReadMode.RGB).type(torch.cuda.FloatTensor), 255.0)
else:
    device = torch.device("cpu")
    data = torch.div(read_image(path, ImageReadMode.RGB).type(torch.FloatTensor), 255.0)

model.to(device)

r = Resize((480,640), antialias=True)
img = r.forward(data).unsqueeze(0)

print(img.size())

img.to(device)

res = model(img)
tr_res = torch.mul(res, 255.0).cpu()

plt.imshow(r.forward(tr_res).squeeze().detach().numpy(), cmap='gray', vmin=0, vmax=255)
plt.show()
