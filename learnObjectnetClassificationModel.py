# -*- coding: utf-8 -*-
"""
Learn models using ObjectNet3D images from setupDataFlipped_objectnet3d
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from axisAngle import get_error2, geodesic_loss
from objectnetHelperFunctions import TestImages, ClassificationModel

import numpy as np
import scipy.io as spio
import gc
import os
import time
import progressbar
import pickle
import argparse
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='Objectnet Models')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--save_str', type=str)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--dict_size', type=int, default=200)
parser.add_argument('--init_lr', type=float, default=1e-4)
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
N0, N1, N2, N3, ndim = 2048, 1000, 500, 100, 3

# paths
db_path = 'data/objectnet'
train_path = os.path.join(db_path, 'train')
test_path = os.path.join(db_path, 'test')

# classes
tmp = spio.loadmat(os.path.join(db_path, 'dbinfo'), squeeze_me=True)
classes = tmp['classes']
num_classes = len(classes)

# kmeans data
kmeans_file = 'data/kmeans_dictionary_axis_angle_' + str(args.dict_size) + '.pkl'
kmeans = pickle.load(open(kmeans_file, 'rb'))
kmeans_dict = kmeans.cluster_centers_
cluster_centers_ = Variable(torch.from_numpy(kmeans_dict).float()).cuda()
num_clusters = kmeans.n_clusters

# loss
mse_loss = nn.MSELoss().cuda()
ce_loss = nn.CrossEntropyLoss().cuda()
gve_loss = geodesic_loss().cuda()

# DATA
# datasets
train_data = TestImages(train_path, classes, args.dict_size)
test_data = TestImages(test_path, classes, args.dict_size)
# setup data loaders
train_loader = DataLoader(train_data, batch_size=96, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32)
print('Train: {0} \t Test: {1}'.format(len(train_loader), len(test_loader)))

# my_model
model = ClassificationModel(num_classes, args.dict_size)
# print(model)
# loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda ep: 1/(1+ep))
# store stuff
writer = SummaryWriter(log_dir)
count = 0
val_loss = []


# OPTIMIZATION functions
def training():
	global count, val_loss
	model.train()
	bar = progressbar.ProgressBar(max_value=len(train_loader))
	for i, sample in enumerate(train_loader):
		# forward steps
		# output
		xdata = Variable(sample['xdata'].cuda())
		label = Variable(sample['label']).cuda()
		ydata = Variable(sample['ydata_bin']).cuda()
		output = model(xdata, label)
		# loss
		loss = ce_loss(output, ydata.squeeze())
		# parameter updates
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		# cleanup
		del output, sample, loss, xdata, ydata
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
		ind = torch.argmax(output, dim=1).data.cpu().numpy()
		ypred.append(kmeans_dict[ind, :])
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


for epoch in range(args.num_epochs):
	tic = time.time()
	# scheduler.step()
	# training step
	training()
	# save model at end of epoch
	save_checkpoint(model_file)
	# validation
	ytest, yhat_test, test_labels = testing()
	tmp = get_error2(ytest, yhat_test, test_labels, num_classes)
	print('\nMedErr: {0}'.format(tmp))
	val_loss.append(tmp)
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
