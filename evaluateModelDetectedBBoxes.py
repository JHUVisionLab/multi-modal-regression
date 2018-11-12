# -*- coding: utf-8 -*-
"""
Evaluate on detected bboxes using Geodesic regression and
Geodesic Bin and Delta models for the axis-angle representation
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from dataGenerators import Dataset, preprocess_real
from binDeltaModels import OneBinDeltaModel, OneDeltaPerBinModel
from featureModels import resnet_model, vgg_model
from poseModels import model_3layer
from helperFunctions import classes

import numpy as np
import scipy.io as spio
import os
import pickle
import argparse
import progressbar

parser = argparse.ArgumentParser(description='Evaluate on Detections')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--save_str', type=str)
parser.add_argument('--dict_size', type=int, default=200)
parser.add_argument('--feature_network', type=str, default='resnet')
parser.add_argument('--N0', type=int, default=2048)
parser.add_argument('--N1', type=int, default=1000)
parser.add_argument('--N2', type=int, default=500)
parser.add_argument('--N3', type=int, default=100)
parser.add_argument('--multires', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model_type', type=str, default='bd')
args = parser.parse_args()
print(args)
# assign GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


class DetImages(Dataset):
	def __init__(self, db_path):
		super().__init__()
		self.db_path = db_path
		self.image_names = []
		tmp = spio.loadmat(os.path.join(self.db_path, 'dbinfo'), squeeze_me=True)
		self.image_names = tmp['image_names']

	def __len__(self):
		return len(self.image_names)

	def __getitem__(self, idx):
		image_name = self.image_names[idx]
		tmp = spio.loadmat(os.path.join(self.db_path, 'all', image_name), verify_compressed_data_integrity=False)
		xdata = tmp['xdata']
		if xdata.size == 0:
			return {'xdata': torch.FloatTensor()}
		xdata = torch.stack([preprocess_real(xdata[i]) for i in range(xdata.shape[0])]).float()
		label = torch.from_numpy(tmp['labels']-1).long()
		bbox = torch.from_numpy(tmp['bboxes']).float()
		sample = {'xdata': xdata, 'label': label, 'bbox': bbox}
		return sample


# relevant variables
ndim = 3
num_classes = len(classes)


# my_model
class RegressionModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_classes = num_classes
		if args.feature_network == 'resnet':
			self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		elif args.feature_network == 'vgg':
			self.feature_model = vgg_model('vgg13', 'fc6').cuda()
		self.pose_models = nn.ModuleList(
			[model_3layer(args.N0, args.N1, args.N2, ndim) for i in range(self.num_classes)]).cuda()

	def forward(self, x, label):
		x = self.feature_model(x)
		x = torch.stack([self.pose_models[i](x) for i in range(self.num_classes)]).permute(1, 2, 0)
		label = torch.zeros(label.size(0), self.num_classes).scatter_(1, label.data.cpu(), 1.0)
		label = Variable(label.unsqueeze(2).cuda())
		y = torch.squeeze(torch.bmm(x, label), 2)
		y = np.pi * F.tanh(y)
		del x, label
		return y


class ClassificationModel(nn.Module):
	def __init__(self, dict_size):
		super().__init__()
		self.num_classes = num_classes
		self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		self.pose_models = nn.ModuleList([model_3layer(args.N0, args.N1, args.N2, dict_size) for i in range(self.num_classes)]).cuda()

	def forward(self, x, label):
		x = self.feature_model(x)
		x = torch.stack([self.pose_models[i](x) for i in range(self.num_classes)]).permute(1, 2, 0)
		label = torch.zeros(label.size(0), self.num_classes).scatter_(1, label.data.cpu(), 1.0)
		label = Variable(label.unsqueeze(2).cuda())
		y = torch.squeeze(torch.bmm(x, label), 2)
		del x, label
		return y


if args.model_type == 'bd' or args.model_type == 'c':
	# kmeans data
	kmeans_file = 'data/kmeans_dictionary_axis_angle_' + str(args.dict_size) + '.pkl'
	kmeans = pickle.load(open(kmeans_file, 'rb'))
	kmeans_dict = kmeans.cluster_centers_
	num_clusters = kmeans.n_clusters

	if args.model_type == 'c':
		model = ClassificationModel(num_clusters)
	else:
		# my_model
		if not args.multires:
			model = OneBinDeltaModel(args.feature_network, num_classes, num_clusters, args.N0, args.N1, args.N2, ndim)
		else:
			model = OneDeltaPerBinModel(args.feature_network, num_classes, num_clusters, args.N0, args.N1, args.N2, args.N3, ndim)
else:
	model = RegressionModel()

# load model
model_file = os.path.join('models', args.save_str + '.tar')
model.load_state_dict(torch.load(model_file))


def testing(det_path):
	test_data = DetImages(det_path)
	model.eval()
	ypred = []
	bbox = []
	labels = []
	bar = progressbar.ProgressBar(max_value=len(test_data))
	for i in range(len(test_data)):
		sample = test_data[i]
		if len(sample['xdata']) == 0:
			ypred.append(np.array([]))
			bbox.append(np.array([]))
			labels.append(np.array([]))
			continue
		xdata = Variable(sample['xdata'].cuda())
		label = Variable(sample['label'].cuda())
		tmp_ypred = []
		tmp_xdata = torch.split(xdata, args.batch_size)
		tmp_label = torch.split(label, args.batch_size)
		for j in range(len(tmp_xdata)):
			output = model(tmp_xdata[j], tmp_label[j])
			if args.model_type == 'bd':
				ypred_bin = np.argmax(output[0].data.cpu().numpy(), axis=1)
				ypred_res = output[1].data.cpu().numpy()
				tmp_ypred.append(kmeans_dict[ypred_bin, :] + ypred_res)
			elif args.model_type == 'c':
				ypred_bin = np.argmax(output.data.cpu().numpy(), axis=1)
				tmp_ypred.append(kmeans_dict[ypred_bin, :])
			else:
				tmp_ypred.append(output.data.cpu().numpy())
			del output
		ypred.append(np.concatenate(tmp_ypred))
		bbox.append(sample['bbox'].numpy())
		labels.append(sample['label'].numpy())
		del xdata, label, sample
		bar.update(i+1)
	return bbox, ypred, labels


# evaluate the model
bbox, ypred, labels = testing('data/vk_dets')
results_file = os.path.join('results', args.save_str + '_vk_dets')
spio.savemat(results_file, {'bbox': bbox, 'ypred': ypred, 'labels': labels})

bbox, ypred, labels = testing('data/r4cnn_dets')
results_file = os.path.join('results', args.save_str + '_r4cnn_dets')
spio.savemat(results_file, {'bbox': bbox, 'ypred': ypred, 'labels': labels})

bbox, ypred, labels = testing('data/maskrcnn_dets')
results_file = os.path.join('results', args.save_str + '_maskrcnn_dets')
spio.savemat(results_file, {'bbox': bbox, 'ypred': ypred, 'labels': labels})

bbox, ypred, labels = testing('data/maskrcnn_dets_nofinetune')
results_file = os.path.join('results', args.save_str + '_maskrcnn_dets_nofinetune')
spio.savemat(results_file, {'bbox': bbox, 'ypred': ypred, 'labels': labels})
