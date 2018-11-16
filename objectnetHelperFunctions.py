import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os
import scipy.io as spio
from PIL import Image
import pickle

from helperFunctions import parse_name, rotation_matrix
from axisAngle import get_y
from featureModels import resnet_model

# for image normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), normalize])


class TrainImages(Dataset):
	def __init__(self, data_path, classes, dict_size=16):
		self.db_path = data_path
		self.classes = classes
		self.num_classes = len(self.classes)
		self.list_image_names = []
		for i in range(self.num_classes):
			tmp = spio.loadmat(os.path.join(self.db_path, self.classes[i] + '_info'), squeeze_me=True)
			image_names = tmp['image_names']
			self.list_image_names.append(image_names)
		self.num_images = np.array([len(self.list_image_names[i]) for i in range(self.num_classes)])
		self.image_names = self.list_image_names
		kmeans_file = 'data/kmeans_dictionary_axis_angle_' + str(dict_size) + '.pkl'
		self.kmeans = pickle.load(open(kmeans_file, 'rb'))

	def __len__(self):
		return np.amax(self.num_images)

	def __getitem__(self, idx):
		# return sample with xdata, ydata, label
		xdata, ydata, label = [], [], []
		for i in range(self.num_classes):
			image_name = self.image_names[i][idx % self.num_images[i]]
			label.append(i*torch.ones(1).long())
			# read image
			img_pil = Image.open(os.path.join(self.db_path, self.classes[i], image_name + '.png'))
			xdata.append(preprocess(img_pil))
			# parse image name to get correponding target
			_, _, az, el, ct, _ = parse_name(image_name)
			R = rotation_matrix(az, el, ct)
			tmpy = get_y(R)
			ydata.append(torch.from_numpy(tmpy).float())
		xdata = torch.stack(xdata)
		ydata = torch.stack(ydata)
		ydata_bin = self.kmeans.predict(ydata.numpy())
		ydata_res = ydata.numpy() - self.kmeans.cluster_centers_[ydata_bin, :]
		ydata_bin = torch.from_numpy(ydata_bin).long()
		ydata_res = torch.from_numpy(ydata_res).float()
		label = torch.stack(label)
		sample = {'xdata': xdata, 'ydata': ydata, 'label': label, 'ydata_bin': ydata_bin, 'ydata_res': ydata_res}
		return sample

	def shuffle_images(self):
		self.image_names = [np.random.permutation(self.list_image_names[i]) for i in range(self.num_classes)]


class TestImages(Dataset):
	def __init__(self, data_path, classes, dict_size=16):
		self.db_path = data_path
		self.classes = classes
		self.num_classes = len(self.classes)
		self.list_image_names = []
		self.list_labels = []
		for i in range(self.num_classes):
			tmp = spio.loadmat(os.path.join(self.db_path, self.classes[i] + '_info'), squeeze_me=True)
			image_names = tmp['image_names']
			self.list_image_names.append(image_names)
			self.list_labels.append(i*np.ones(len(image_names), dtype='int'))
		self.image_names = np.concatenate(self.list_image_names)
		self.labels = np.concatenate(self.list_labels)
		kmeans_file = 'data/kmeans_dictionary_axis_angle_' + str(dict_size) + '.pkl'
		self.kmeans = pickle.load(open(kmeans_file, 'rb'))

	def __len__(self):
		return len(self.image_names)

	def __getitem__(self, idx):
		# return sample with xdata, ydata, label
		image_name = self.image_names[idx]
		label = self.labels[idx]
		# read image
		img_pil = Image.open(os.path.join(self.db_path, self.classes[label], image_name + '.png'))
		xdata = preprocess(img_pil)
		# parse image name to get correponding target
		_, _, az, el, ct, _ = parse_name(image_name)
		R = rotation_matrix(az, el, ct)
		tmpy = get_y(R)
		ydata_bin = np.squeeze(self.kmeans.predict(np.expand_dims(tmpy,0)))
		ydata_res = tmpy - self.kmeans.cluster_centers_[ydata_bin, :]
		ydata_bin = ydata_bin*torch.ones(1).long()
		ydata_res = torch.from_numpy(ydata_res).float()
		ydata = torch.from_numpy(tmpy).float()
		label = label*torch.ones(1).long()
		sample = {'xdata': xdata, 'ydata': ydata, 'label': label, 'ydata_bin': ydata_bin, 'ydata_res': ydata_res}
		return sample


class bin_3layer(nn.Module):
	def __init__(self, n0, n1, n2, num_clusters):
		super().__init__()
		self.fc1 = nn.Linear(n0, n1, bias=False)
		self.bn1 = nn.BatchNorm1d(n1)
		self.fc2 = nn.Linear(n1, n2, bias=False)
		self.bn2 = nn.BatchNorm1d(n2)
		self.fc3 = nn.Linear(n2, num_clusters)

	def forward(self, x):
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.fc2(x)))
		x = self.fc3(x)
		return x


class res_3layer(nn.Module):
	def __init__(self, n0, n1, n2, dim):
		super().__init__()
		self.fc1 = nn.Linear(n0, n1, bias=False)
		self.bn1 = nn.BatchNorm1d(n1)
		self.fc2 = nn.Linear(n1, n2, bias=False)
		self.bn2 = nn.BatchNorm1d(n2)
		self.fc3 = nn.Linear(n2, dim)

	def forward(self, x):
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.fc2(x)))
		x = self.fc3(x)
		return x


class res_2layer(nn.Module):
	def __init__(self, n0, n1, dim):
		super().__init__()
		self.fc1 = nn.Linear(n0, n1, bias=False)
		self.bn1 = nn.BatchNorm1d(n1)
		self.fc2 = nn.Linear(n1, dim)

	def forward(self, x):
		x = F.relu(self.bn1(self.fc1(x)))
		x = self.fc2(x)
		return x


class OneBinDeltaModel(nn.Module):
	def __init__(self, num_classes, dict_size=200, n0=2048, n1=1000, n2=500, dim=3):
		super().__init__()
		self.num_classes = num_classes
		self.num_clusters = dict_size
		self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		self.bin_model = bin_3layer(n0+num_classes, n1, n2, self.num_clusters).cuda()
		self.res_model = res_3layer(n0+num_classes, n1, n2, dim).cuda()

	def forward(self, x, label):
		x = self.feature_model(x)
		label = torch.zeros(label.size(0), self.num_classes).scatter_(1, label.data.cpu(), 1.0)
		label = Variable(label.cuda())
		x = torch.cat((x, label), dim=1)
		y1 = self.bin_model(x)
		y2 = self.res_model(x)
		del x
		return [y1, y2]


class OneDeltaPerBinModel(nn.Module):
	def __init__(self, num_classes, dict_size=16, n0=2048, n1=1000, n2=500, n3=100, dim=3):
		super().__init__()
		self.ndim = dim
		self.num_classes = num_classes
		self.num_clusters = dict_size
		self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		self.bin_model = bin_3layer(n0+num_classes, n1, n2, self.num_clusters).cuda()
		self.res_models = nn.ModuleList([res_2layer(n0+num_classes, n3, dim) for i in range(self.num_clusters)]).cuda()

	def forward(self, x, label):
		x = self.feature_model(x)
		label = torch.zeros(label.size(0), self.num_classes).scatter_(1, label.data.cpu(), 1.0)
		label = Variable(label.cuda())
		x = torch.cat((x, label), dim=1)
		y1 = self.bin_model(x)
		y2 = torch.stack([self.res_models[i](x) for i in range(self.num_clusters)])
		y2 = y2.view(self.num_clusters, -1, self.ndim).permute(1, 2, 0)
		pose_label = torch.argmax(y1, dim=1, keepdim=True)
		pose_label = torch.zeros(pose_label.size(0), self.num_clusters).scatter_(1, pose_label.data.cpu(), 1.0)
		pose_label = Variable(pose_label.unsqueeze(2).cuda())
		y2 = torch.squeeze(torch.bmm(y2, pose_label), 2)
		del x, pose_label
		return [y1, y2]


class RegressionModel(nn.Module):
	def __init__(self, num_classes, n0=2048, n1=1000, n2=500, dim=3):
		super().__init__()
		self.num_classes = num_classes
		self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		self.pose_model = res_3layer(n0+num_classes, n1, n2, dim).cuda()

	def forward(self, x, label):
		x = self.feature_model(x)
		label = torch.zeros(label.size(0), self.num_classes).scatter_(1, label.data.cpu(), 1.0)
		label = Variable(label.cuda())
		x = torch.cat((x, label), dim=1)
		x = self.pose_model(x)
		x = np.pi * F.tanh(x)
		return x


class ClassificationModel(nn.Module):
	def __init__(self, num_classes, dict_size=16, n0=2048, n1=1000, n2=500):
		super().__init__()
		self.num_classes = num_classes
		self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		self.pose_model = bin_3layer(n0+num_classes, n1, n2, dict_size).cuda()

	def forward(self, x, label):
		x = self.feature_model(x)
		label = torch.zeros(label.size(0), self.num_classes).scatter_(1, label.data.cpu(), 1.0)
		label = Variable(label.cuda())
		x = torch.cat((x, label), dim=1)
		x = self.pose_model(x)
		return x
