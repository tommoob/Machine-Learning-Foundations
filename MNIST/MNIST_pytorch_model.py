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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.float()

        x = self.pool(self.relu(self.conv1(x)))  # Output: (batch_size, 32, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))  # Output: (batch_size, 64, 7, 7)
        x = x.view(-1, 64 * 7 * 7)               # Flatten the tensor, Output: (batch_size, 64 * 7 * 7)
        x = self.relu(self.fc1(x))               # Output: (batch_size, 128)
        x = self.dropout(x)
        x = self.fc2(x)                          # Output: (batch_size, 10)
        return x