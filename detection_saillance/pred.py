import torch
import model4 as mod
from torchvision.io import read_image
from torchvision.transforms import Resize
from torchvision.io import ImageReadMode
import matplotlib.pyplot as plt

model  = mod.Net()
model.load_state_dict(torch.load("./results/model4l1.pth"))
model.eval()

path = "./dataset/data/images/val/COCO_val2014_000000016439.jpg"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    data = torch.subtract(torch.div(read_image(path, ImageReadMode.RGB).type(torch.cuda.FloatTensor), 255.0), 0.2)
else:
    device = torch.device("cpu")
    data = torch.subtract(torch.div(read_image(path, ImageReadMode.RGB).type(torch.FloatTensor), 255.0), 0.2)

model.to(device)

r = Resize((480,640), antialias=True)
img = r.forward(data).unsqueeze(0)

print(img.size())

img.to(device)

res = model(img)
tr_res = torch.add(torch.mul(res, 255.0), 0.2).cpu()

plt.imshow(r.forward(tr_res).squeeze().detach().numpy())
plt.show()
