# -*- coding: utf-8 -*-
"""
Independent model based on Geodesic Bin and Delta model M_G+
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataGenerators import TestImages, my_collate
from binDeltaGenerators import GBDGenerator
from binDeltaModels import bin_3layer, res_2layer, resnet_model
from axisAngle import get_error2, geodesic_loss
from helperFunctions import classes, get_accuracy

import numpy as np
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
parser.add_argument('--render_path', type=str, default='data/renderforcnn/')
parser.add_argument('--augmented_path', type=str, default='data/augmented2/')
parser.add_argument('--pascal3d_path', type=str, default='data/flipped_new/test/')
parser.add_argument('--save_str', type=str)
parser.add_argument('--dict_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--feature_network', type=str, default='resnet')
parser.add_argument('--N0', type=int, default=2048)
parser.add_argument('--N1', type=int, default=1000)
parser.add_argument('--N2', type=int, default=500)
parser.add_argument('--N3', type=int, default=100)
parser.add_argument('--init_lr', type=float, default=1e-4)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--max_iterations', type=float, default=np.inf)
args = parser.parse_args()
print(args)
# assign GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# save stuff here
results_file = os.path.join('results', args.save_str)
model_file = os.path.join('models', args.save_str + '.tar')
plots_file = os.path.join('plots', args.save_str)
log_dir = os.path.join('logs', args.save_str)

# kmeans data
kmeans_file = 'data/kmeans_dictionary_axis_angle_' + str(args.dict_size) + '.pkl'
kmeans = pickle.load(open(kmeans_file, 'rb'))
kmeans_dict = kmeans.cluster_centers_
cluster_centers_ = Variable(torch.from_numpy(kmeans_dict).float()).cuda()
num_clusters = kmeans.n_clusters

# relevant variables
ndim = 3
num_classes = len(classes)

# loss
mse_loss = nn.MSELoss().cuda()
ce_loss = nn.CrossEntropyLoss().cuda()
gve_loss = geodesic_loss().cuda()

# DATA
# datasets
real_data = GBDGenerator(args.augmented_path, 'real', kmeans_file)
render_data = GBDGenerator(args.render_path, 'render', kmeans_file)
test_data = TestImages(args.pascal3d_path)
# setup data loaders
real_loader = DataLoader(real_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
render_loader = DataLoader(render_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
test_loader = DataLoader(test_data, batch_size=32)
print('Real: {0} \t Render: {1} \t Test: {2}'.format(len(real_loader), len(render_loader), len(test_loader)))

if np.isinf(args.max_iterations):
	max_iterations = min(len(real_loader), len(render_loader))
else:
	max_iterations = args.max_iterations


# my_model
class OneDeltaPerBinModel(nn.Module):
	def __init__(self, feature_network, num_classes, num_clusters, N0, N1, N2, N3, ndim):
		super().__init__()
		self.num_classes = num_classes
		self.num_clusters = num_clusters
		self.ndim = ndim
		self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		self.bin_model = bin_3layer(N0, N1, N2, num_clusters).cuda()
		self.res_models = nn.ModuleList([res_2layer(N0, N3, ndim) for i in range(self.num_clusters)]).cuda()

	def forward(self, x):
		x = self.feature_model(x)
		y1 = self.bin_model(x)
		y2 = torch.stack([self.res_models[i](x) for i in range(self.num_clusters)])
		y2 = y2.view(self.num_clusters, -1, self.ndim).permute(1, 2, 0)
		pose_label = torch.argmax(y1, dim=1, keepdim=True)
		pose_label = torch.zeros(pose_label.size(0), self.num_clusters).scatter_(1, pose_label.data.cpu(), 1.0)
		pose_label = Variable(pose_label.unsqueeze(2).cuda())
		y2 = torch.squeeze(torch.bmm(y2, pose_label), 2)
		del x, pose_label
		return [y1, y2]     # [pose_bin, pose_delta]


model = OneDeltaPerBinModel(args.feature_network, num_classes, num_clusters, args.N0, args.N1, args.N2, args.N3, ndim)
# print(model)
# loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
# store stuff
writer = SummaryWriter(log_dir)
count = 0
val_loss = []


# OPTIMIZATION functions
def training_init():
	global count, val_loss
	model.train()
	bar = progressbar.ProgressBar(max_value=max_iterations)
	for i, (sample_real, sample_render) in enumerate(zip(real_loader, render_loader)):
		# forward steps
		# outputs
		xdata_real = Variable(sample_real['xdata'].cuda())
		ydata_bin_real = Variable(sample_real['ydata_bin'].cuda())
		ydata_res_real = Variable(sample_real['ydata_res'].cuda())
		output_real = model(xdata_real)
		xdata_render = Variable(sample_render['xdata'].cuda())
		ydata_bin_render = Variable(sample_render['ydata_bin'].cuda())
		ydata_res_render = Variable(sample_render['ydata_res'].cuda())
		output_render = model(xdata_render)
		# loss
		ydata_bin = torch.cat((ydata_bin_real, ydata_bin_render))
		ydata_res = torch.cat((ydata_res_real, ydata_res_render))
		output_bin = torch.cat((output_real[0], output_render[0]))
		output_res = torch.cat((output_real[1], output_render[1]))
		Lc = ce_loss(output_bin, ydata_bin)
		Lr = mse_loss(output_res, ydata_res)
		loss = Lc + Lr
		# parameter updates
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		if i % 1000 == 0:
			ytest, yhat_test, test_labels = testing()
			spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
			tmp_val_loss = get_error2(ytest, yhat_test, test_labels, num_classes)
			writer.add_scalar('val_loss', tmp_val_loss, count)
			val_loss.append(tmp_val_loss)
		# cleanup
		del xdata_real, xdata_render, ydata_bin_real, ydata_bin_render
		del ydata_bin, ydata_res, output_bin, output_res, ydata_res_real, ydata_res_render
		del output_real, output_render, loss, sample_real, sample_render, Lr, Lc
		bar.update(i)
		# stop
		if i == max_iterations:
			break
	render_loader.dataset.shuffle_images()
	real_loader.dataset.shuffle_images()


def training():
	global count, val_loss
	model.train()
	bar = progressbar.ProgressBar(max_value=max_iterations)
	for i, (sample_real, sample_render) in enumerate(zip(real_loader, render_loader)):
		# forward steps
		# output
		xdata_real = Variable(sample_real['xdata'].cuda())
		ydata_bin_real = Variable(sample_real['ydata_bin'].cuda())
		ydata_real = Variable(sample_real['ydata'].cuda())
		output_real = model(xdata_real)
		xdata_render = Variable(sample_render['xdata'].cuda())
		ydata_bin_render = Variable(sample_render['ydata_bin'].cuda())
		ydata_render = Variable(sample_render['ydata'].cuda())
		output_render = model(xdata_render)
		# loss
		ydata_bin = torch.cat((ydata_bin_real, ydata_bin_render))
		ydata = torch.cat((ydata_real, ydata_render))
		output_bin = torch.cat((output_real[0], output_render[0]))
		ind = torch.argmax(output_bin, dim=1)
		y = torch.index_select(cluster_centers_, 0, ind)
		output = y + torch.cat((output_real[1], output_render[1]))
		Lc = ce_loss(output_bin, ydata_bin)
		Lr = gve_loss(output, ydata)
		loss = Lc + 10*Lr
		# parameter updates
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		if i % 1000 == 0:
			ytest, yhat_test, test_labels = testing()
			spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
			tmp_val_loss = get_error2(ytest, yhat_test, test_labels, num_classes)
			writer.add_scalar('val_loss', tmp_val_loss, count)
			val_loss.append(tmp_val_loss)
		# cleanup
		del xdata_real, xdata_render, ydata_bin_real, ydata_bin_render
		del ydata_bin, ydata, output_bin, output, ydata_real, ydata_render
		del output_real, output_render, loss, sample_real, sample_render, Lr, Lc
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
		output = model(xdata)
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


def save_checkpoint(filename):
	torch.save(model.state_dict(), filename)


# initialization
training_init()
ytest, yhat_test, test_labels = testing()
print('\nMedErr: {0}'.format(get_error2(ytest, yhat_test, test_labels, num_classes)))

for epoch in range(args.num_epochs):
	tic = time.time()
	scheduler.step()
	# training step
	training()
	# save model at end of epoch
	save_checkpoint(model_file)
	# validation
	ytest, yhat_test, test_labels = testing()
	print('\nMedErr: {0}'.format(get_error2(ytest, yhat_test, test_labels, num_classes)))
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
