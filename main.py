from data import get_data
from model import BreastCancerClassifier
from train import train_model
from evaluate import evaluate_model

import torch
import torch.nn as nn
import torch.optim as optim

train_loader, test_loader = get_data()

model = BreastCancerClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer)

evaluate_model(model, test_loader)

torch.save(model.state_dict(), 'breast_cancer_classifier.pth')