import torch
import torch.nn as nn

"""
This module defines neural network architectures for stability prediction. The MLP class implements a multi-layer perceptron with three fully connected layers (128-64-1 neurons), ReLU activations and dropout regularization (0.25 rate), and sigmoid output for binary classification. The make_mlp factory function creates models with configurable input dimensions. The architecture is designed for tabular feature inputs from the dynamics extractors, balancing expressiveness with regularization to prevent overfitting on limited simulation data. It assumes PyTorch is available and GPU support is optional.
"""

class MLP(nn.Module):
	def __init__(self, input_dim):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(input_dim, 128)
		self.dropout1 = nn.Dropout(0.25)
		self.fc2 = nn.Linear(128, 64)
		self.dropout2 = nn.Dropout(0.25)
		self.fc3 = nn.Linear(64, 1)
		
	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = self.dropout1(x)
		x = torch.relu(self.fc2(x))
		x = self.dropout2(x)
		x = self.fc3(x)
		return x


def make_mlp(input_dim):
	return MLP(input_dim)
