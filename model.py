import torch
import torch.nn as nn


class BreastCancerClassifier(nn.Module):
    def __init__(self):
        super(BreastCancerClassifier, self).__init__()
        self.fc1 = nn.Linear(30, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)  # Output layer with 2 neurons

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here since we'll use CrossEntropyLoss
        return x
