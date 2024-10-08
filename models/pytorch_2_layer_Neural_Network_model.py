# -*- coding: utf-8 -*-
"""
model.ipynb

#Load Functions
"""

import torch
import torch.nn as nn

"""#Model Function"""

# Define Neural Network
class PricePredictionModel(nn.Module):
    def __init__(self,input_size):
        super(PricePredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def initialize_model(input_size):
    """
    Initialize the PricePredictionModel with the given input size.

    Args:
        input_size (int): Number of input features.

    Returns:
        model (PricePredictionModel): Initialized model.
    """
    model = PricePredictionModel(input_size)
    return model

'''
usage eample:
from model import initialize_model

input_size = X_train.shape[1]
model = initialize_model(input_size).to(device)
'''
