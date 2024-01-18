import torch
import torch.optim as optim
import premade
import os
import torch.nn.functional as F
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from torchvision.io import ImageReadMode

batch_size_test = 8

random_seed = 12
# random_seed = 13
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

class SaliconImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.labels_dir = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])
        image = torch.div(read_image(img_path, ImageReadMode.RGB).type(torch.cuda.FloatTensor), 255.0)
        label_path = os.path.join(self.labels_dir, os.listdir(self.labels_dir)[idx])
        label = read_image(label_path, ImageReadMode.GRAY)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = torch.div(self.target_transform(label).type(torch.cuda.FloatTensor), 255.0)
        return image, label

test_loader = torch.utils.data.DataLoader(
  SaliconImageDataset('dataset/truth/val/', 'dataset/data/images/val/', transform=Resize((480,640), antialias=True),target_transform=Resize((30,40), antialias=True)),
  batch_size=batch_size_test, shuffle=True)

loss = torch.nn.BCELoss()
network = premade.Net()
network.load_state_dict(torch.load('./results/premade.pth'))
network.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network.to(device)

if not os.path.isdir('./results'):
  os.mkdir('./results')

def test(device):
    network.eval()
    lossq = 0
    i = 0
    with torch.no_grad():
       for data, target in test_loader:
            data.to(device)
            output = network(data)
            target.to(device)
            lossq += loss(output, target)
            i += 1
    lossq /= i
    print('\nTest set: Avg. loss: {:.4f}\n'.format(
        lossq))

test(device)