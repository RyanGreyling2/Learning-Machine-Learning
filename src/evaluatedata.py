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
        # print("NUM OF FEATURES: ", np.shape(self.features))
        self.scenario = data[::,-1].astype('int64')

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # sample = {'features': self.features[idx], 'target': self.target[idx]}
        sample = {'features': self.features[idx], 'scenario': self.scenario[idx]}
        # print(np.shape(sample['features']), np.shape(sample['target']))
        # print("===")
        return sample

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

evaluate_dataset = TrainData(xlsx_file='../modified-data/evaluatefeatures.xlsx')

net = Net()
net.load_state_dict(torch.load('../networks/evaluate.pth'))

evaluations = []
for i in range(evaluate_dataset.__len__()):
    data_point = evaluate_dataset.__getitem__(i)
    input = torch.from_numpy(data_point['features'])
    scenario_num = data_point['scenario']

    output = net(input)
    result = 1

    if output[0].item() > output[1].item():
        result = 0
    
    evaluations.append([scenario_num, result])

evaluations = pd.DataFrame(data=evaluations, columns=['dataset_id', 'prediction_score'])
evaluations.to_excel("../modified-data/final_submission.xlsx", startcol=-1)
