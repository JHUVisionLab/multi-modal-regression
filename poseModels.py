from torch import nn
import torch.nn.functional as F

"""
Pose Models
"""


# 3 fully connected layerss
class model_3layer(nn.Module):

	def __init__(self, N0, N1, N2, N3):
		super().__init__()
		self.fc1 = nn.Linear(N0, N1, bias=False)
		self.bn1 = nn.BatchNorm1d(N1)
		self.fc2 = nn.Linear(N1, N2, bias=False)
		self.bn2 = nn.BatchNorm1d(N2)
		self.fc3 = nn.Linear(N2, N3)

	def forward(self, x):
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.fc2(x)))
		x = self.fc3(x)
		return x


# 2 fully connected layers
class model_2layer(nn.Module):

	def __init__(self, N0, N1, N2):
		super().__init__()
		self.fc1 = nn.Linear(N0, N1, bias=False)
		self.bn1 = nn.BatchNorm1d(N1)
		self.fc2 = nn.Linear(N1, N2)

	def forward(self, x):
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.tanh(self.fc2(x))
		return x


# 1 fully connected layer
class model_1layer(nn.Module):

	def __init__(self, N0, N1):
		super().__init__()
		self.fc = nn.Linear(N0, N1)

	def forward(self, x):
		x = self.fc(x)
		return x


