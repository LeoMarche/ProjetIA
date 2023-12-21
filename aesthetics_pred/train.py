import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import model
import os
import torch.nn.functional as F
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from torchvision.io import ImageReadMode

n_epochs = 3
batch_size_train = 32
batch_size_test = 1000
learning_rate = 0.0001
momentum = 0.5
log_interval = 10

random_seed = 12
# random_seed = 13
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

class AADBDDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.label_csv = pd.read_csv(annotations_file, delimiter=",")
        self.labels_dir = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label_csv)

    def __getitem__(self, idx):
        label = self.label_csv.iloc[idx]['score']
        img_file = self.label_csv.iloc[idx]['ImageFile']
        img_path = os.path.join(self.img_dir, img_file)
        image = torch.div(read_image(img_path, ImageReadMode.RGB).type(torch.cuda.FloatTensor), 255.0)
        
        if self.transform:
            image = self.transform(image)
        return image, torch.cuda.FloatTensor([label])

train_loader = torch.utils.data.DataLoader(
  AADBDDataset('dataset/data/Dataset.csv', 'dataset/data/datasetImages_warp256/datasetImages_warp256', transform=Resize((256,256), antialias=True)),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  AADBDDataset('dataset/data/Dataset_test.csv', 'dataset/data/datasetImages_warp256/datasetImages_warp256', transform=Resize((256,256), antialias=True)),
  batch_size=batch_size_train, shuffle=True)

loss = torch.nn.MSELoss()
rankloss = torch.nn.MarginRankingLoss(0.01)
network = model.Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1)

if not os.path.isdir('./results'):
  os.mkdir('./results')

def train(epoch, device):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    data.to(device)
    output = network(data)
    output_rank = torch.cartesian_prod(output.squeeze(), output.squeeze())
    x1 = output_rank[:,0]
    x2 = output_rank[:,1]
    target.to(device)
    rank = torch.cartesian_prod(target.squeeze(), target.squeeze())
    target_rank = torch.ones_like(x1).to(device).to(x1.dtype)
    target_rank[rank[:,0] < rank[:,1]] = -1.0
    lossq = loss(output, target)
    ranklossq = rankloss(x1, x2, target_rank)
    if batch_idx % log_interval == 0:
      torch.save(network.state_dict(), './results/model2.pth')
      torch.save(optimizer.state_dict(), './results/optimizermodel2.pth')
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRank Loss: {:.6f}'.format(
        epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), lossq.item(), ranklossq.item()))
    (lossq+ranklossq).backward()
    optimizer.step()

def test(device):
  network.eval()
  test_loss = 0
  i = 0
  with torch.no_grad():
    for data, target in test_loader:
      data.to(device)
      output = network(data)
      target.to(device)
      test_loss += loss(output, target)
      i += 1
  test_loss /= i
  print('\nTest set: Avg. loss: {:.4f}\n'.format(
    test_loss))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

network.to(device)

for epoch in range(1, n_epochs + 1):
  test(device)
  train(epoch, device)
  scheduler.step()

test(device)