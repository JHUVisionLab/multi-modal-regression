# -*- coding: utf-8 -*-
"""
Geodesic Bin and Delta model for the axis-angle representation
"""

import torch
from torch.autograd import Variable

from dataGenerators import Dataset, preprocess_real
from binDeltaModels import OneBinDeltaModel, OneDeltaPerBinModel
from helperFunctions import classes

import numpy as np
import scipy.io as spio
import os
import pickle
import argparse

parser = argparse.ArgumentParser(description='Geodesic Bin & Delta Model')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--det_type', type=str, default='r4cnn')
parser.add_argument('--save_str', type=str)
parser.add_argument('--dict_size', type=int, default=200)
parser.add_argument('--feature_network', type=str, default='resnet')
parser.add_argument('--N0', type=int, default=2048)
parser.add_argument('--N1', type=int, default=1000)
parser.add_argument('--N2', type=int, default=500)
parser.add_argument('--N3', type=int, default=100)
parser.add_argument('--multires', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()
print(args)
# assign GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

if args.det_type == 'r4cnn':
	det_path = 'data/r4cnn_dets/'
elif args.det_type == 'vk':
	det_path = 'data/vk_dets/'
elif args.det_type == 'maskrcnn':
	det_path = 'data/maskrcnn_dets/'
elif args.det_type == 'maskrcnn_nofinetune':
	det_path = 'data/maskrcnn_dets_nofinetune'
else:
	raise NameError('Unknown det_type passed')


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
			return {'xdata': torch.from_numpy(np.array([]))}
		xdata = torch.stack([preprocess_real(xdata[i]) for i in range(xdata.shape[0])]).float()
		label = torch.from_numpy(tmp['labels']-1).long()
		bbox = torch.from_numpy(tmp['bboxes']).float()
		sample = {'xdata': xdata, 'label': label, 'bbox': bbox}
		return sample


test_data = DetImages(det_path)

# save stuff here
model_file = os.path.join('models', args.save_str + '.tar')
results_file = os.path.join('results', args.save_str + '_' + args.det_type + '_dets')

# kmeans data
kmeans_file = 'data/kmeans_dictionary_axis_angle_' + str(args.dict_size) + '.pkl'
kmeans = pickle.load(open(kmeans_file, 'rb'))
kmeans_dict = kmeans.cluster_centers_
num_clusters = kmeans.n_clusters

# relevant variables
ndim = 3
num_classes = len(classes)

# my_model
if not args.multires:
	model = OneBinDeltaModel(args.feature_network, num_classes, num_clusters, args.N0, args.N1, args.N2, ndim)
else:
	model = OneDeltaPerBinModel(args.feature_network, num_classes, num_clusters, args.N0, args.N1, args.N2, args.N3, ndim)
# load model
model.load_state_dict(torch.load(model_file))


def testing():
	model.eval()
	ypred = []
	bbox = []
	labels = []
	for i in range(len(test_data)):
		print(i)
		sample = test_data[i]
		if sample['xdata'].size == 0:
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
			ypred_bin = np.argmax(output[0].data.cpu().numpy(), axis=1)
			ypred_res = output[1].data.cpu().numpy()
			tmp_ypred.append(kmeans_dict[ypred_bin, :] + ypred_res)
			del output
		ypred.append(np.concatenate(tmp_ypred))
		bbox.append(sample['bbox'].numpy())
		labels.append(sample['label'].numpy())
		del xdata, label, sample
	return bbox, ypred, labels


# evaluate the model
bbox, ypred, labels = testing()
spio.savemat(results_file, {'bbox': bbox, 'ypred': ypred, 'labels': labels})
