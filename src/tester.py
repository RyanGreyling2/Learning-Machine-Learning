import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random

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

test_dataset = TrainData(xlsx_file='../modified-data/featureselected.xlsx')

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

PATH = '../networks/evaluate.pth'
net = Net()
net.load_state_dict(torch.load(PATH))

print("Testing:")

total_positives = 0
true_positives = 0
false_positives = 0

true_rand_positives = 0
false_rand_positives = 0

for i in range(3000):
    data_point = test_dataset.__getitem__(i)
    input = torch.from_numpy(data_point['features'])
    #input = torch.Tensor(list(data_point['features']))
    target = data_point['target']

    if target==1:
        total_positives += 1

    # print(input)
    # print(np.shape(input))

    output = net(input)
    result = 1

    if output[0].item() > output[1].item():
        result = 0

    if result == 1 and target == 1:
        true_positives += 1
    elif result == 1:
        false_positives += 1
    
    rand_result = random.randint(0,1)

    if rand_result == 1 and target == 1:
        true_rand_positives += 1
    elif rand_result == 1:
        false_rand_positives += 1

if true_positives != 0:
    inverse_recall = total_positives/true_positives
    inverse_precision = (true_positives + false_positives)/true_positives
    F1 = 2/(inverse_recall + inverse_precision)
else:
    F1 = 0

percentage = (3000 - total_positives - false_positives + true_positives)/30

rand_percentage = (3000 - total_positives - false_rand_positives + true_rand_positives)/30

inverse_rand_recall = total_positives/true_rand_positives
inverse_rand_precision = (true_rand_positives + false_rand_positives)/true_rand_positives
randF1 = 2/(inverse_rand_recall + inverse_rand_precision)

print("Our model got %", percentage,"correct")

print("Aw Geez, our model got a F1 score of ", F1, " out of 1")

print("Random Guessing got %", rand_percentage," correct")

print("Random Guessing had a F1 score of ", randF1, " out of 1")