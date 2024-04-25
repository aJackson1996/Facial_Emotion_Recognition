# This is a sample Python script.
from torch import nn
import torch.nn.functional as F
from torchvision.ops import RoIPool


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#less layers, first two layers have larger filters
class FER_Model(nn.Module):
    def __init__(self):
        super(FER_Model, self).__init__()
        self.Conv1 = nn.Conv2d(1, 64, 5, padding=2, stride=1)
        self.Conv1_Normalize = nn.BatchNorm2d(64)
        self.Pooling = nn.MaxPool2d(2)
        self.Conv2 = nn.Conv2d(64, 128, 5, padding=2, stride=1)
        self.Conv2_Normalize = nn.BatchNorm2d(128)
        self.Conv3 = nn.Conv2d(128, 256, 3, padding=1, stride=1)
        self.Conv3_Normalize = nn.BatchNorm2d(256)
        self.Conv4 = nn.Conv2d(256, 512, 3, padding=1, stride=1)
        self.Conv4_Normalize = nn.BatchNorm2d(512)
        self.Conv5 = nn.Conv2d(512, 1024, 3, padding=1, stride=1)
        self.Conv5_Normalize = nn.BatchNorm2d(1024)
        self.fc1 = nn.Linear(1024, 800)
        self.fc2 = nn.Linear(800, 1600)
        self.output = nn.Linear(1600, 7)
        self.dropout = nn.Dropout(p = .3)

    def forward(self, X):
        X = F.relu(self.Conv1(X))
        X = self.Conv1_Normalize(X)
        X = self.Pooling(X)
        X = F.relu(self.Conv2(X))
        X = self.Conv2_Normalize(X)
        X = self.Pooling(X)
        X = F.relu(self.Conv3(X))
        X = self.Conv3_Normalize(X)
        X = self.Pooling(X)
        X = F.relu(self.Conv4(X))
        X = self.Conv4_Normalize(X)
        X = self.Pooling(X)
        X = F.relu(self.Conv5(X))
        X = self.Conv5_Normalize(X)
        X = self.Pooling(X)
        X = X.reshape(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = F.relu(self.fc2(X))
        X = self.dropout(X)
        return F.softmax(self.output(X), dim=1)


#improved model, where the layers with larger filters were replaced with two layers with smaller filters
class FER_Model_v2(nn.Module):
    def __init__(self):
        super(FER_Model_v2, self).__init__()
        self.Conv1_1 = nn.Conv2d(1, 64, 3, padding=1, stride=1)
        self.Conv1_2 = nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.Conv1_Normalize = nn.BatchNorm2d(64)
        self.Pooling = nn.MaxPool2d(2)
        self.Conv2_1 = nn.Conv2d(64, 128, 3, padding=1, stride=1)
        self.Conv2_2 = nn.Conv2d(128, 128, 3, padding=1, stride=1)
        self.Conv2_Normalize = nn.BatchNorm2d(128)
        self.Conv3 = nn.Conv2d(128, 256, 3, padding=1, stride=1)
        self.Conv3_Normalize = nn.BatchNorm2d(256)
        self.Conv4 = nn.Conv2d(256, 512, 3, padding=1, stride=1)
        self.Conv4_Normalize = nn.BatchNorm2d(512)
        self.Conv5 = nn.Conv2d(512, 1024, 3, padding=1, stride=1)
        self.Conv5_Normalize = nn.BatchNorm2d(1024)
        self.fc1 = nn.Linear(1024, 800)
        self.fc2 = nn.Linear(800, 1600)
        self.output = nn.Linear(1600, 7)
        self.dropout = nn.Dropout(p = .3)

    def forward(self, X):
        X = F.relu(self.Conv1_1(X))
        X = F.relu(self.Conv1_2(X))
        X = self.Conv1_Normalize(X)
        X = self.Pooling(X)
        X = F.relu(self.Conv2_1(X))
        X = F.relu(self.Conv2_2(X))
        X = self.Conv2_Normalize(X)
        X = self.Pooling(X)
        X = F.relu(self.Conv3(X))
        X = self.Conv3_Normalize(X)
        X = self.Pooling(X)
        X = F.relu(self.Conv4(X))
        X = self.Conv4_Normalize(X)
        X = self.Pooling(X)
        X = F.relu(self.Conv5(X))
        X = self.Conv5_Normalize(X)
        X = self.Pooling(X)
        X = X.reshape(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = F.relu(self.fc2(X))
        X = self.dropout(X)
        return F.softmax(self.output(X), dim=1)
