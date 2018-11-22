# -*- coding: utf-8 -*-
"""
Learn models using ObjectNet3D images from setupDataFlipped_objectnet3d
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from quaternion import get_error2, geodesic_loss, get_y
from objectnetHelperFunctions import OneBinDeltaModel, OneDeltaPerBinModel
from helperFunctions import parse_name, rotation_matrix

import numpy as np
import math
import scipy.io as spio
import gc
import os
import time
import progressbar
import argparse
from tensorboardX import SummaryWriter
from PIL import Image


parser = argparse.ArgumentParser(description='Objectnet Models')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--save_str', type=str)
parser.add_argument('--dict_size', type=int, default=200)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--multires', type=bool, default=False)
parser.add_argument('--init_lr', type=float, default=1e-4)
parser.add_argument('--db_path', type=str, default='data/objectnet3d/flipped')
args = parser.parse_args()
print(args)
# assign GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# save stuff here
results_file = os.path.join('results', args.save_str)
model_file = os.path.join('models', args.save_str + '.tar')
plots_file = os.path.join('plots', args.save_str)
log_dir = os.path.join('logs', args.save_str)

# constants
N0, N1, N2, N3, ndim = 2048, 1000, 500, 100, 4

# paths
train_path = os.path.join(args.db_path, 'train')
test_path = os.path.join(args.db_path, 'test')

# classes
tmp = spio.loadmat(os.path.join(args.db_path, 'dbinfo'), squeeze_me=True)
classes = tmp['classes']
num_classes = len(classes)

# pose dictionary
s = 1/math.sqrt(2)
kmeans_dict = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                        [s, s, 0, 0], [s, 0, s, 0], [s, 0, 0, s], [0, s, s, 0],
                        [0, s, 0, s], [0, 0, s, s], [s, -s, 0, 0], [s, 0, -s, 0],
                        [s, 0, 0, -s], [0, s, -s, 0], [0, s, 0, -s], [0, 0, s, -s]])
cluster_centers_ = Variable(torch.from_numpy(kmeans_dict).float()).cuda()
num_clusters = 16

# loss
mse_loss = nn.MSELoss().cuda()
ce_loss = nn.CrossEntropyLoss().cuda()
gve_loss = geodesic_loss().cuda()


# DATA
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), normalize])


class TestImages(Dataset):
	def __init__(self, data_path, classes):
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
		ydata_bin = np.argmax(np.abs(np.dot(kmeans_dict, tmpy)))
		ydata_res = tmpy - kmeans_dict[ydata_bin, :]
		ydata_bin = ydata_bin*torch.ones(1).long()
		ydata_res = torch.from_numpy(ydata_res).float()
		ydata = torch.from_numpy(tmpy).float()
		label = label*torch.ones(1).long()
		sample = {'xdata': xdata, 'ydata': ydata, 'label': label, 'ydata_bin': ydata_bin, 'ydata_res': ydata_res}
		return sample


# datasets
train_data = TestImages(train_path, classes)
test_data = TestImages(test_path, classes)
# setup data loaders
train_loader = DataLoader(train_data, batch_size=96, num_workers=4, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32)
print('Train: {0} \t Test: {1}'.format(len(train_loader), len(test_loader)))

# my_model
if not args.multires:
	model = OneBinDeltaModel(num_classes)
else:
	model = OneDeltaPerBinModel(num_classes)
# print(model)
# loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda ep: (10**-(ep//10))/(1+ep%10))
# store stuff
writer = SummaryWriter(log_dir)
count = 0
val_loss = []


# OPTIMIZATION functions
def training_init():
	global count
	model.train()
	bar = progressbar.ProgressBar(max_value=len(train_loader))
	for i, sample in enumerate(train_loader):
		# forward steps
		# outputs
		xdata = Variable(sample['xdata'].cuda())
		label = Variable(sample['label']).cuda()
		ydata_bin = Variable(sample['ydata_bin']).cuda().squeeze()
		ydata_res = Variable(sample['ydata_res']).cuda()
		output = model(xdata, label)
		# loss
		Lc = ce_loss(output[0], ydata_bin)
		Lr = mse_loss(output[1], ydata_res)
		loss = Lc + Lr
		# parameter updates
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		# cleanup
		del xdata, ydata_bin, ydata_res, output, loss, Lc, Lr
		bar.update(i+1)


def training():
	global count
	model.train()
	bar = progressbar.ProgressBar(max_value=len(train_loader))
	for i, sample in enumerate(train_loader):
		# forward steps
		# output
		xdata = Variable(sample['xdata'].cuda())
		label = Variable(sample['label']).cuda()
		ydata_bin = Variable(sample['ydata_bin']).cuda().squeeze()
		ydata = Variable(sample['ydata']).cuda()
		output = model(xdata, label)
		# loss
		ind = torch.argmax(output[0], dim=1)
		y = torch.index_select(cluster_centers_, 0, ind) + output[1]
		Lc = ce_loss(output[0], ydata_bin)
		Lr = gve_loss(y, ydata)
		loss = Lc + 10*Lr
		# parameter updates
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		# cleanup
		del xdata, ydata_bin, ydata, output, y, Lr, Lc, loss, ind
		bar.update(i+1)


def testing():
	model.eval()
	ypred = []
	ytrue = []
	labels = []
	bar = progressbar.ProgressBar(max_value=len(test_loader))
	for i, sample in enumerate(test_loader):
		xdata = Variable(sample['xdata'].cuda())
		label = Variable(sample['label'].cuda())
		output = model(xdata, label)
		ypred_bin = np.argmax(output[0].data.cpu().numpy(), axis=1)
		ypred_res = output[1].data.cpu().numpy()
		y = kmeans_dict[ypred_bin, :] + ypred_res
		ypred.append(y / np.maximum(np.linalg.norm(y, 2, 1, True), 1e-10))
		ytrue.append(sample['ydata'].numpy())
		labels.append(sample['label'].numpy())
		del xdata, label, output, sample
		gc.collect()
		bar.update(i+1)
	ypred = np.concatenate(ypred)
	ytrue = np.concatenate(ytrue)
	labels = np.concatenate(labels)
	model.train()
	return ytrue, ypred, labels


def save_checkpoint(filename):
	torch.save(model.state_dict(), filename)


# initialization
training_init()
ytest, yhat_test, test_labels = testing()
print('\nMedErr: {0}'.format(get_error2(ytest, yhat_test, test_labels, num_classes)))

s = 0  # reset
for epoch in range(args.num_epochs):
	tic = time.time()
	scheduler.step()
	# training step
	training()
	# save model at end of epoch
	save_checkpoint(model_file)
	# validation
	ytest, yhat_test, test_labels = testing()
	tmp = get_error2(ytest, yhat_test, test_labels, num_classes)
	val_loss.append(tmp)
	print('\nMedErr: {0}'.format(tmp))
	spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
	# time and output
	toc = time.time() - tic
	print('Epoch: {0} done in time {1}s'.format(epoch, toc))
	# cleanup
	gc.collect()
writer.close()
val_loss = np.stack(val_loss)
spio.savemat(plots_file, {'val_loss': val_loss})

# evaluate the model
ytest, yhat_test, test_labels = testing()
print('\nMedErr: {0}'.format(get_error2(ytest, yhat_test, test_labels, num_classes)))
spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
