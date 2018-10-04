# -*- coding: utf-8 -*-
"""
Geodesic Bin and Delta model for the axis-angle representation
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataGenerators import TestImages, my_collate
from binDeltaGenerators import GBDGenerator
from axisAngle import get_error2, geodesic_loss
from binDeltaModels import OneBinDeltaModel, OneDeltaPerBinModel
from helperFunctions import classes, mySGD

import numpy as np
import math
import scipy.io as spio
import gc
import os
import time
import progressbar
import pickle
import argparse
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Geodesic Bin & Delta Model')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--save_str', type=str)
parser.add_argument('--dict_size', type=int, default=200)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--feature_network', type=str, default='resnet')
parser.add_argument('--num_epochs', type=int, default=9)
parser.add_argument('--multires', type=bool, default=False)
parser.add_argument('--db_type', type=str, default='clean')
args = parser.parse_args()
print(args)
# assign GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# save stuff here
model_file = os.path.join('models', args.save_str + '.tar')
results_dir = os.path.join('results', args.save_str + '_' + args.db_type)
plots_file = os.path.join('plots', args.save_str + '_' + args.db_type)
log_dir = os.path.join('logs', args.save_str + '_' + args.db_type)
if not os.path.exists(results_dir):
	os.mkdir(results_dir)

# kmeans data
kmeans_file = 'data/kmeans_dictionary_axis_angle_' + str(args.dict_size) + '.pkl'
kmeans = pickle.load(open(kmeans_file, 'rb'))
kmeans_dict = kmeans.cluster_centers_
cluster_centers_ = Variable(torch.from_numpy(kmeans_dict).float()).cuda()
num_clusters = kmeans.n_clusters

# relevant variables
ndim = 3
N0, N1, N2, N3 = 2048, 1000, 500, 100
num_classes = len(classes)
if args.db_type == 'clean':
	db_path = 'data/flipped_new'
else:
	db_path = 'data/flipped_all'
num_classes = len(classes)
train_path = os.path.join(db_path, 'train')
test_path = os.path.join(db_path, 'test')
render_path = 'data/renderforcnn/'


ce_loss = nn.CrossEntropyLoss().cuda()
gve_loss = geodesic_loss().cuda()

# DATA
# datasets
real_data = GBDGenerator(train_path, 'real', kmeans_file)
render_data = GBDGenerator(render_path, 'render', kmeans_file)
test_data = TestImages(test_path)
# setup data loaders
real_loader = DataLoader(real_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
render_loader = DataLoader(render_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
test_loader = DataLoader(test_data, batch_size=32)
print('Real: {0} \t Render: {1} \t Test: {2}'.format(len(real_loader), len(render_loader), len(test_loader)))

max_iterations = len(real_loader)

# my_model
if not args.multires:
	model = OneBinDeltaModel(args.feature_network, num_classes, num_clusters, N0, N1, N2, ndim)
else:
	model = OneDeltaPerBinModel(args.feature_network, num_classes, num_clusters, N0, N1, N2, N3, ndim)
model.load_state_dict(torch.load(model_file))
# print(model)
optimizer = mySGD(model.parameters(), c=2*len(real_loader))
writer = SummaryWriter(log_dir)
count = 0
val_loss = []
s = 0
num_ensemble = 0


def training():
	global count, val_loss, s, num_ensemble
	model.train()
	bar = progressbar.ProgressBar(max_value=max_iterations)
	for i, (sample_real, sample_render) in enumerate(zip(real_loader, render_loader)):
		# forward steps
		# output
		xdata_real = Variable(sample_real['xdata'].cuda())
		label_real = Variable(sample_real['label'].cuda())
		ydata_real = [Variable(sample_real['ydata_bin'].cuda()), Variable(sample_real['ydata'].cuda())]
		output_real = model(xdata_real, label_real)
		xdata_render = Variable(sample_render['xdata'].cuda())
		label_render = Variable(sample_render['label'].cuda())
		ydata_render = [Variable(sample_render['ydata_bin'].cuda()), Variable(sample_render['ydata'].cuda())]
		output_render = model(xdata_render, label_render)
		# loss
		ydata_bin = torch.cat((ydata_real[0], ydata_render[0]))
		ydata = torch.cat((ydata_real[1], ydata_render[1]))
		output_bin = torch.cat((output_real[0], output_render[0]))
		_, ind = torch.max(output_bin, dim=1)
		y = torch.index_select(cluster_centers_, 0, ind)
		output = y + torch.cat((output_real[1], output_render[1]))
		Lc = ce_loss(output_bin, ydata_bin)
		Lr = gve_loss(output, ydata)
		loss = Lc + math.exp(-s)*Lr + s
		# parameter updates
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		s = math.log(Lr)
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		writer.add_scalar('alpha', math.exp(-s), count)
		if i % 500 == 0:
			ytest, yhat_test, test_labels = testing()
			tmp_val_loss = get_error2(ytest, yhat_test, test_labels, num_classes)
			writer.add_scalar('val_loss', tmp_val_loss, count)
			val_loss.append(tmp_val_loss)
		count += 1
		if count % optimizer.c == optimizer.c / 2:
			ytest, yhat_test, test_labels = testing()
			num_ensemble += 1
			results_file = os.path.join(results_dir, 'num' + str(num_ensemble))
			spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
		# cleanup
		del xdata_real, xdata_render, label_real, label_render, ydata_real, ydata_render
		del ydata_bin, ydata, output_bin, output
		del output_real, output_render, sample_real, sample_render, loss
		bar.update(i)
		# stop
		if i == max_iterations:
			break
	render_loader.dataset.shuffle_images()
	real_loader.dataset.shuffle_images()


def testing():
	model.eval()
	ypred = []
	ytrue = []
	labels = []
	for i, sample in enumerate(test_loader):
		xdata = Variable(sample['xdata'].cuda())
		label = Variable(sample['label'].cuda())
		output = model(xdata, label)
		ypred_bin = np.argmax(output[0].data.cpu().numpy(), axis=1)
		ypred_res = output[1].data.cpu().numpy()
		ypred.append(kmeans_dict[ypred_bin, :] + ypred_res)
		ytrue.append(sample['ydata'].numpy())
		labels.append(sample['label'].numpy())
		del xdata, label, output, sample
		gc.collect()
	ypred = np.concatenate(ypred)
	ytrue = np.concatenate(ytrue)
	labels = np.concatenate(labels)
	model.train()
	return ytrue, ypred, labels


ytest, yhat_test, test_labels = testing()
print('\nMedErr: {0}'.format(get_error2(ytest, yhat_test, test_labels, num_classes)))
results_file = os.path.join(results_dir, 'num'+str(num_ensemble))
spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})

for epoch in range(args.num_epochs):
	tic = time.time()
	# training step
	training()
	# validation
	ytest, yhat_test, test_labels = testing()
	tmp_val_loss = get_error2(ytest, yhat_test, test_labels, num_classes)
	print('\nMedErr: {0}'.format(tmp_val_loss))
	writer.add_scalar('val_loss', tmp_val_loss, count)
	val_loss.append(tmp_val_loss)
	# time and output
	toc = time.time() - tic
	print('Epoch: {0} done in time {1}s'.format(epoch, toc))
	# cleanup
	gc.collect()
writer.close()
val_loss = np.stack(val_loss)
spio.savemat(plots_file, {'val_loss': val_loss})
