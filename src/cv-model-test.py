import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
from bettertester import *

init_arr = list(range(1, 3001))
random.shuffle(init_arr)

first_half = pd.read_excel('../modified-data/featureselected.xlsx', skiprows=init_arr[0:1500])
second_half = pd.read_excel('../modified-data/featureselected.xlsx', skiprows=init_arr[1500:3000])
# Create 2 new spreadsheets. 1) Based on first 1500 indices of 

first_half.to_excel("../temp-data/first_half.xlsx")
second_half.to_excel("../temp-data/second_half.xlsx")

class TrainData(Dataset):

    def __init__(self, xlsx_file):
        data = pd.read_excel(xlsx_file)
        data = np.array(data)
        # data = np.delete(data, -2, 1)
        self.features = data[::,1:-1:].astype('float32')
        self.target = data[::,-1].astype('int64')

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        sample = {'features': self.features[idx], 'target': self.target[idx]}
        # print(np.shape(sample['features']), np.shape(sample['target']))
        # print("===")
        return sample

# class Net(nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 3x3 square convolution
#         # kernel
#         # self.conv1 = nn.Conv2d(1, 6, 3)
#         # self.conv2 = nn.Conv2d(6, 16, 3)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(30, 16)  # 6*6 from image dimension
#         self.fc2 = nn.Linear(16, 8)
#         self.fc3 = nn.Linear(8, 2)

#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square you can only specify a single number
#         # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         # x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = torch.sigmoid(self.fc3(x))
#         return x

batch_size = 1
train_size = 100

train_dataset = TrainData(xlsx_file='../temp-data/first_half.xlsx')
trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

for epoch in range(train_size):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data["features"]
        labels = data["target"]

        # print("inputs", np.shape(inputs))
        # print("labels", np.shape(labels))

        # print("=====")

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        # print("outputs", np.shape(outputs))
        # print("labels", np.shape(labels))

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

torch.save(net.state_dict(), '../networks/firsthalf.pth')

print("Testing First Half:")
first_F1 = CalculateF1('../networks/firsthalf.pth', '../temp-data/second_half.xlsx')

train_dataset = TrainData(xlsx_file='../temp-data/second_half.xlsx')
trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

for epoch in range(train_size):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data["features"]
        labels = data["target"]

        # print("inputs", np.shape(inputs))
        # print("labels", np.shape(labels))

        # print("=====")

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        # print("outputs", np.shape(outputs))
        # print("labels", np.shape(labels))

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

torch.save(net.state_dict(), '../networks/secondhalf.pth')

print("Testing Second Half:")
second_F1 = CalculateF1('../networks/secondhalf.pth', '../temp-data/first_half.xlsx')

print("First Half F1 was ", first_F1)
print("Second Half F1 was ", second_F1)
print("Average F1 is ", (first_F1 + second_F1)/2)