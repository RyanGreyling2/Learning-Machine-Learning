import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd   
import numpy as np

# trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

class TrainData(Dataset):

    def __init__(self, xlsx_file):
        data = pd.read_excel(xlsx_file)
        data = np.array(data)
        data = np.delete(data, -2, 1)
        self.data = data.astype('float32')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {'features': self.data[::,:-1:], 'target': self.data[::,-1]}
        return sample

batch_size = 100

train_dataset = TrainData(xlsx_file='../original-data/Training-Data.xlsx')
trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        # self.conv1 = nn.Conv2d(1, 6, 3)
        # self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(30, 16)  # 6*6 from image dimension
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data["features"]
        labels = data["target"]

        print("inputs", np.shape(inputs))
        print("labels", np.shape(labels))

        print("=====")

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        print("outputs", np.shape(outputs))
        print("labels", np.shape(labels))

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)