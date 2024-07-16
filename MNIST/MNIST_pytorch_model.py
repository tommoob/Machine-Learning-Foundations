import torch.nn as nn
import torch
import torch.nn.functional as F


class MNIST_hidden_layer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MNIST_hidden_layer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x.to(torch.float32)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(32, 128, 3, 1)
        self.conv2 = nn.Conv2d(128, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(144, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output