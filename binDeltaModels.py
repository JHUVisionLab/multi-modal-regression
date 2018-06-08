# -*- coding: utf-8 -*-
"""
Bin and Delta models
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from featureModels import resnet_model, vgg_model

"""
Building Blocks
"""


class bin_1layer(nn.Module):
	def __init__(self, N0, num_clusters):
		super().__init__()
		self.fc = nn.Linear(N0, num_clusters)

	def forward(self, x):
		x = self.fc(x)
		return x


class res_1layer(nn.Module):
	def __init__(self, N0, ndim):
		super().__init__()
		self.fc = nn.Linear(N0, ndim)

	def forward(self, x):
		x = self.fc(x)
		return x


class bin_2layer(nn.Module):
	def __init__(self, N0, N1, num_clusters):
		super().__init__()
		self.fc1 = nn.Linear(N0, N1, bias=False)
		self.bn1 = nn.BatchNorm1d(N1)
		self.fc2 = nn.Linear(N1, num_clusters)

	def forward(self, x):
		x = F.relu(self.bn1(self.fc1(x)))
		x = self.fc2(x)
		return x


class res_2layer(nn.Module):
	def __init__(self, N0, N1, ndim):
		super().__init__()
		self.fc1 = nn.Linear(N0, N1, bias=False)
		self.bn1 = nn.BatchNorm1d(N1)
		self.fc2 = nn.Linear(N1, ndim)

	def forward(self, x):
		x = F.relu(self.bn1(self.fc1(x)))
		x = self.fc2(x)
		return x


class bin_3layer(nn.Module):
	def __init__(self, N0, N1, N2, num_clusters):
		super().__init__()
		self.fc1 = nn.Linear(N0, N1, bias=False)
		self.bn1 = nn.BatchNorm1d(N1)
		self.fc2 = nn.Linear(N1, N2, bias=False)
		self.bn2 = nn.BatchNorm1d(N2)
		self.fc3 = nn.Linear(N2, num_clusters)

	def forward(self, x):
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.fc2(x)))
		x = self.fc3(x)
		return x


class res_3layer(nn.Module):
	def __init__(self, N0, N1, N2, ndim):
		super().__init__()
		self.fc1 = nn.Linear(N0, N1, bias=False)
		self.bn1 = nn.BatchNorm1d(N1)
		self.fc2 = nn.Linear(N1, N2, bias=False)
		self.bn2 = nn.BatchNorm1d(N2)
		self.fc3 = nn.Linear(N2, ndim)

	def forward(self, x):
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.fc2(x)))
		x = self.fc3(x)
		return x


"""
Bin and Delta models for different settings
"""


class OneBinDeltaModel(nn.Module):
	def __init__(self, feature_network, num_classes, num_clusters, N0, N1, N2, ndim):
		super().__init__()
		self.num_classes = num_classes
		self.num_clusters = num_clusters
		self.ndim = ndim
		if feature_network == 'resnet':
			self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		elif feature_network == 'vgg':
			self.feature_model = vgg_model('vgg13', 'fc6').cuda()
		self.bin_models = nn.ModuleList([bin_3layer(N0, N1, N2, num_clusters) for i in range(self.num_classes)]).cuda()
		self.res_models = nn.ModuleList([res_3layer(N0, N1, N2, ndim) for i in range(self.num_classes)]).cuda()

	def forward(self, x, label):
		x = self.feature_model(x)
		y1 = torch.stack([self.bin_models[i](x) for i in range(self.num_classes)]).permute(1, 2, 0)
		y2 = torch.stack([self.res_models[i](x) for i in range(self.num_classes)]).permute(1, 2, 0)
		label = torch.zeros(label.size(0), self.num_classes).scatter_(1, label.data.cpu(), 1.0)
		label = Variable(label.unsqueeze(2).cuda())
		y1 = torch.squeeze(torch.bmm(y1, label), 2)
		y2 = torch.squeeze(torch.bmm(y2, label), 2)
		del x, label
		return [y1, y2]


class OneDeltaPerBinModel(nn.Module):
	def __init__(self, feature_network, num_classes, num_clusters, N0, N1, N2, N3, ndim):
		super().__init__()
		self.num_classes = num_classes
		self.num_clusters = num_clusters
		self.ndim = ndim
		if feature_network == 'resnet':
			self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		elif feature_network == 'vgg':
			self.feature_model = vgg_model('vgg13', 'fc6').cuda()
		self.bin_models = nn.ModuleList([bin_3layer(N0, N1, N2, num_clusters) for i in range(self.num_classes)]).cuda()
		self.res_models = nn.ModuleList([res_2layer(N0, N3, ndim) for i in range(self.num_classes * self.num_clusters)]).cuda()

	def forward(self, x, class_label):
		x = self.feature_model(x)
		y1 = torch.stack([self.bin_models[i](x) for i in range(self.num_classes)]).permute(1, 2, 0)
		y2 = torch.stack([self.res_models[i](x) for i in range(self.num_classes * self.num_clusters)])
		y2 = y2.view(self.num_classes, self.num_clusters, -1, self.ndim).permute(1, 2, 3, 0)
		class_label = torch.zeros(class_label.size(0), self.num_classes).scatter_(1, class_label.data.cpu(), 1.0)
		class_label = Variable(class_label.unsqueeze(2).cuda())
		y1 = torch.squeeze(torch.bmm(y1, class_label), 2)
		y2 = torch.squeeze(torch.matmul(y2, class_label), 3)
		_, pose_label = torch.max(y1, dim=1, keepdim=True)
		pose_label = torch.zeros(pose_label.size(0), self.num_clusters).scatter_(1, pose_label.data.cpu(), 1.0)
		pose_label = Variable(pose_label.unsqueeze(2).cuda())
		y2 = torch.squeeze(torch.bmm(y2.permute(1, 2, 0), pose_label), 2)
		del x, class_label, pose_label
		return [y1, y2]


