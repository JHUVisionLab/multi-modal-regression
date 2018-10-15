# -*- coding: utf-8 -*-
"""
Joint Cat & Pose model (Top1) with Geodesic Bin and Delta model for the axis-angle representation
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataGenerators import TestImages
from binDeltaModels import OneBinDeltaModel, OneDeltaPerBinModel

import numpy as np
import scipy.io as spio
import gc
import os
import pickle
import argparse

parser = argparse.ArgumentParser(description='Geodesic Bin & Delta Model')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--save_str', type=str)
parser.add_argument('--dict_size', type=int, default=200)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--feature_network', type=str, default='resnet')
parser.add_argument('--multires', type=bool, default=False)
parser.add_argument('--db_type', type=str, default='clean')
args = parser.parse_args()
print(args)
# assign GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# save stuff here
oracale_model_file = os.path.join('models', args.save_str + '.tar')
cat_model_file = os.path.join('models', args.save_str + '_cat.tar')
model_file = os.path.join('models', args.save_str + '_top1.tar')
results_file = os.path.join('results', args.save_str + '_top1_' + args.db_type + '_analysis')

# kmeans data
kmeans_file = 'data/kmeans_dictionary_axis_angle_' + str(args.dict_size) + '.pkl'
kmeans = pickle.load(open(kmeans_file, 'rb'))
kmeans_dict = kmeans.cluster_centers_
cluster_centers_ = Variable(torch.from_numpy(kmeans_dict).float()).cuda()
num_clusters = kmeans.n_clusters

# relevant variables
ndim, num_classes = 3, 12
N0, N1, N2, N3 = 2048, 1000, 500, 100
if args.db_type == 'clean':
	db_path = 'data/flipped_new'
else:
	db_path = 'data/flipped_all'
test_path = os.path.join(db_path, 'test')

# DATA
test_data = TestImages(test_path)
test_loader = DataLoader(test_data, batch_size=32)

# my_model
if not args.multires:
	orig_model = OneBinDeltaModel(args.feature_network, num_classes, num_clusters, N0, N1, N2, ndim)
else:
	orig_model = OneDeltaPerBinModel(args.feature_network, num_classes, num_clusters, N0, N1, N2, N3, ndim)


class JointCatPoseModel(nn.Module):
	def __init__(self, oracle_model):
		super().__init__()
		# old stuff
		self.num_classes = oracle_model.num_classes
		self.num_clusters = oracle_model.num_clusters
		self.ndim = oracle_model.ndim
		self.feature_model = oracle_model.feature_model
		self.bin_models = oracle_model.bin_models
		self.res_models = oracle_model.res_models
		# new stuff
		self.fc = nn.Linear(N0, num_classes).cuda()

	def forward(self, x):
		x = self.feature_model(x)
		y0 = self.fc(x)
		ypred = []
		for i in range(self.num_classes):
			ybin = self.bin_models[i](x)
			ind = torch.argmax(ybin, dim=1)
			if not args.multires:
				yres = self.res_models[i](x)
			else:
				pose_label = torch.zeros(ind.size(0), self.num_clusters).scatter_(1, ind.data.cpu(), 1.0)
				pose_label = Variable(pose_label.unsqueeze(2).cuda())
				yres = []
				for j in range(self.num_clusters):
					yres.append(self.res_models[i * self.num_clusters + j](x))
				yres = torch.stack(yres).permute(1, 2, 0)
				yres = torch.squeeze(torch.bmm(yres, pose_label), 2)
				del pose_label
			y = cluster_centers_.index_select(0, ind) + yres
			ypred.append(y)
		y1 = torch.stack(ypred).permute(1, 2, 0)
		del ypred, ybin, ind, yres, y
		return [y0, y1]   # cat, pose


orig_model.load_state_dict(torch.load(oracale_model_file))
model = JointCatPoseModel(orig_model)
model.eval()


def testing():
	ytrue_cat, ytrue_pose = [], []
	ypred_cat, ypred_pose = [], []
	for i, sample in enumerate(test_loader):
		xdata = Variable(sample['xdata'].cuda())
		output = model(xdata)
		output_cat = output[0].data.cpu().numpy()
		output_pose = output[1].data.cpu().numpy()
		print(i, output_cat.shape, output_pose.shape)
		tmp_labels = np.argmax(output_cat, axis=1)
		ypred_cat.append(tmp_labels)
		ytrue_cat.append(sample['label'].squeeze().numpy())
		ypred_pose.append(output_pose)
		ytrue_pose.append(sample['ydata'].numpy())
		del xdata, output, sample, output_cat, output_pose, tmp_labels
		gc.collect()
	ytrue_cat = np.concatenate(ytrue_cat)
	ypred_cat = np.concatenate(ypred_cat)
	ytrue_pose = np.concatenate(ytrue_pose)
	ypred_pose = np.concatenate(ypred_pose)
	return ytrue_cat, ytrue_pose, ypred_cat, ypred_pose


ytrue_cat, ytrue_pose, ypred_cat, ypred_pose = testing()
pose_results = {'ytrue_cat': ytrue_cat, 'ytrue_pose': ytrue_pose, 'ypred_cat': ypred_cat, 'ypred_pose': ypred_pose}

model.load_state_dict(torch.load(cat_model_file))
ytrue_cat, ytrue_pose, ypred_cat, ypred_pose = testing()
cat_results = {'ytrue_cat': ytrue_cat, 'ytrue_pose': ytrue_pose, 'ypred_cat': ypred_cat, 'ypred_pose': ypred_pose}

model.load_state_dict(torch.load(model_file))
ytrue_cat, ytrue_pose, ypred_cat, ypred_pose = testing()
joint_results = {'ytrue_cat': ytrue_cat, 'ytrue_pose': ytrue_pose, 'ypred_cat': ypred_cat, 'ypred_pose': ypred_pose}

spio.savemat(results_file, {'pose_results': pose_results, 'cat_results': cat_results, 'joint_results': joint_results})

