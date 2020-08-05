import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
  [transforms.ToTensor(),
  transforms.Normalize((0.5), (0.5))
])

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True)

dataiter = iter(testloader)
images, labels = dataiter.next()

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# print images
print('GroundTruth: ', ' '.join('%s' % classes[labels[j]] for j in range(4)))
imshow(torchvision.utils.make_grid(images))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) #26 x 26
        self.pool = nn.MaxPool2d(2, 2) #24 x 24
        self.conv2 = nn.Conv2d(6, 16, 3) #22 x 22
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

PATH = './cifar_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))

print("Printing Predictions:")

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
