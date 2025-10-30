import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, x1=32, m1=3, x2=64, m2=3, x3=128, d=0.5, K=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, x1, m1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(x1, x2, m2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(x2 * 5 * 5, x3)
        self.dropout = nn.Dropout(d)
        self.fc2 = nn.Linear(x3, K)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
