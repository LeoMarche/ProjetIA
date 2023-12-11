import torch
import torch.optim as optim
import premade
import os
import torch.nn.functional as F
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from torchvision.io import ImageReadMode

n_epochs = 1
batch_size_train = 8
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

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

train_loader = torch.utils.data.DataLoader(
  SaliconImageDataset('dataset/truth/train/', 'dataset/data/images/train/', transform=Resize((480,640), antialias=True),target_transform=Resize((30,40), antialias=True)),
  batch_size=batch_size_train, shuffle=True)

loss = torch.nn.BCELoss()
network = premade.Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

if not os.path.isdir('./results'):
  os.mkdir('./results')

def train(epoch, device):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    data.to(device)
    output = network(data)
    # output = torch.max(output, torch.zeros(output.size(), device=device) + 0.000001)
    # output = torch.div(output, torch.sum(output))
    target.to(device)
    # target = torch.max(target, torch.zeros(target.size(), device=device) + 0.000001)
    # target = torch.div(target, torch.sum(target))
    #loss = F.kl_div(torch.log(output), torch.log(target), reduction='batchmean', log_target=True)
    lossq = loss(output, target)
    if batch_idx % log_interval == 0:
      torch.save(network.state_dict(), './results/premade.pth')
      torch.save(optimizer.state_dict(), './results/optimizerpremade.pth')
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), lossq.item()))
    lossq.backward()
    optimizer.step()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

network.to(device)

for epoch in range(1, n_epochs + 1):
  train(epoch, device)
  scheduler.step()