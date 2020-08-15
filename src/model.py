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


class TrainData(Dataset):

    def __init__(self, xlsx_file):
        data = pd.read_excel(xlsx_file)
        data = np.array(data)
        # data = np.delete(data, -2, 1)
        self.features = data[::,:-1:].astype('float32')
        print("NUM OF FEATURES: ", np.shape(self.features))
        self.target = data[::,-1].astype('int64')

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        sample = {'features': self.features[idx], 'target': self.target[idx]}
        # print(np.shape(sample['features']), np.shape(sample['target']))
        # print("===")
        return sample

batch_size = 1

train_dataset = TrainData(xlsx_file='../modified-data/featureselected.xlsx')
trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(16, 2)
        # self.fc2 = nn.Linear(11, 7)
        # self.fc3 = nn.Linear(7, 4)
        # self.fc4 = nn.Linear(4, 2)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # # x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc1(x))
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

for epoch in range(100):  # loop over the dataset multiple times

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

print('Finished Training')

PATH = '../networks/evaluate.pth'
torch.save(net.state_dict(), PATH)