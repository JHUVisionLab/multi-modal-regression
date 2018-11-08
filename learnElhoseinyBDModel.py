# -*- coding: utf-8 -*-
"""
Elhoseiny model based on Geodesic Bin and Delta model M_G+
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
		self.category_model = nn.Linear(N0, num_classes).cuda()

	def forward(self, x):
		x = self.feature_model(x)
		y0 = self.category_model(x)
		y1 = self.bin_model(x)
		y2 = torch.stack([self.res_models[i](x) for i in range(self.num_clusters)])
		y2 = y2.view(self.num_clusters, -1, self.ndim).permute(1, 2, 0)
		pose_label = torch.argmax(y1, dim=1, keepdim=True)
		pose_label = torch.zeros(pose_label.size(0), self.num_clusters).scatter_(1, pose_label.data.cpu(), 1.0)
		pose_label = Variable(pose_label.unsqueeze(2).cuda())
		y2 = torch.squeeze(torch.bmm(y2, pose_label), 2)
		del x, pose_label
		return [y0, y1, y2]     # [cat, pose_bin, pose_delta]


model = OneDeltaPerBinModel(args.feature_network, num_classes, num_clusters, args.N0, args.N1, args.N2, args.N3, ndim)
# print(model)
# loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
# store stuff
writer = SummaryWriter(log_dir)
count = 0
val_err, val_acc = [], []


# OPTIMIZATION functions
def training_init():
	global count, val_err, val_acc
	model.train()
	bar = progressbar.ProgressBar(max_value=max_iterations)
	for i, (sample_real, sample_render) in enumerate(zip(real_loader, render_loader)):
		# forward steps
		# outputs
		xdata_real = Variable(sample_real['xdata'].cuda())
		label_real = Variable(sample_real['label'].cuda())
		ydata_bin_real = Variable(sample_real['ydata_bin'].cuda())
		ydata_res_real = Variable(sample_real['ydata_res'].cuda())
		output_real = model(xdata_real)
		xdata_render = Variable(sample_render['xdata'].cuda())
		label_render = Variable(sample_render['label'].cuda())
		ydata_bin_render = Variable(sample_render['ydata_bin'].cuda())
		ydata_res_render = Variable(sample_render['ydata_res'].cuda())
		output_render = model(xdata_render)
		# loss
		ydata_bin = torch.cat((ydata_bin_real, ydata_bin_render))
		ydata_res = torch.cat((ydata_res_real, ydata_res_render))
		output_bin = torch.cat((output_real[1], output_render[1]))
		output_res = torch.cat((output_real[2], output_render[2]))
		Lc_cat = ce_loss(output_real[0], label_real.squeeze())
		Lc = ce_loss(output_bin, ydata_bin)
		Lr = mse_loss(output_res, ydata_res)
		loss = Lc_cat + Lc + Lr
		# parameter updates
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		if i % 1000 == 0:
			ytrue_cat, ytrue_pose, ypred_cat, ypred_pose = testing()
			spio.savemat(results_file, {'ytrue_cat': ytrue_cat, 'ytrue_pose': ytrue_pose, 'ypred_cat': ypred_cat, 'ypred_pose': ypred_pose})
			tmp_acc = get_accuracy(ytrue_cat, ypred_cat, num_classes)
			tmp_err = get_error2(ytrue_pose, ypred_pose, ytrue_cat, num_classes)
			writer.add_scalar('val_acc', tmp_acc, count)
			writer.add_scalar('val_err', tmp_err, count)
			val_acc.append(tmp_acc)
			val_err.append(tmp_err)
		# cleanup
		del xdata_real, xdata_render, label_real, label_render, ydata_bin_real, ydata_bin_render
		del ydata_bin, ydata_res, output_bin, output_res, ydata_res_real, ydata_res_render
		del output_real, output_render, loss, sample_real, sample_render, Lr, Lc, Lc_cat
		bar.update(i)
		# stop
		if i == max_iterations:
			break
	render_loader.dataset.shuffle_images()
	real_loader.dataset.shuffle_images()


def training():
	global count, val_err, val_acc
	model.train()
	bar = progressbar.ProgressBar(max_value=max_iterations)
	for i, (sample_real, sample_render) in enumerate(zip(real_loader, render_loader)):
		# forward steps
		# output
		xdata_real = Variable(sample_real['xdata'].cuda())
		label_real = Variable(sample_real['label'].cuda())
		ydata_bin_real = Variable(sample_real['ydata_bin'].cuda())
		ydata_real = Variable(sample_real['ydata'].cuda())
		output_real = model(xdata_real)
		xdata_render = Variable(sample_render['xdata'].cuda())
		label_render = Variable(sample_render['label'].cuda())
		ydata_bin_render = Variable(sample_render['ydata_bin'].cuda())
		ydata_render = Variable(sample_render['ydata'].cuda())
		output_render = model(xdata_render)
		# loss
		ydata_bin = torch.cat((ydata_bin_real, ydata_bin_render))
		ydata = torch.cat((ydata_real, ydata_render))
		output_bin = torch.cat((output_real[1], output_render[1]))
		ind = torch.argmax(output_bin, dim=1)
		y = torch.index_select(cluster_centers_, 0, ind)
		output = y + torch.cat((output_real[2], output_render[2]))
		Lc_cat = ce_loss(output_real[0], label_real.squeeze())
		Lc = ce_loss(output_bin, ydata_bin)
		Lr = gve_loss(output, ydata)
		loss = 0.1*Lc_cat + Lc + 10*Lr
		# parameter updates
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		if i % 1000 == 0:
			ytrue_cat, ytrue_pose, ypred_cat, ypred_pose = testing()
			spio.savemat(results_file, {'ytrue_cat': ytrue_cat, 'ytrue_pose': ytrue_pose, 'ypred_cat': ypred_cat, 'ypred_pose': ypred_pose})
			tmp_acc = get_accuracy(ytrue_cat, ypred_cat, num_classes)
			tmp_err = get_error2(ytrue_pose, ypred_pose, ytrue_cat, num_classes)
			writer.add_scalar('val_acc', tmp_acc, count)
			writer.add_scalar('val_err', tmp_err, count)
			val_acc.append(tmp_acc)
			val_err.append(tmp_err)
		# cleanup
		del xdata_real, xdata_render, label_real, label_render, ydata_bin_real, ydata_bin_render
		del ydata_bin, ydata, output_bin, output, ydata_real, ydata_render
		del output_real, output_render, loss, sample_real, sample_render, Lr, Lc, Lc_cat
		bar.update(i)
		# stop
		if i == max_iterations:
			break
	render_loader.dataset.shuffle_images()
	real_loader.dataset.shuffle_images()


def testing():
	model.eval()
	ytrue_cat, ytrue_pose = [], []
	ypred_cat, ypred_pose = [], []
	for i, sample in enumerate(test_loader):
		xdata = Variable(sample['xdata'].cuda())
		output = model(xdata)
		output_cat = output[0]
		output_bin = output[1]
		output_res = output[2]
		tmp_labels = np.argmax(output_cat.data.cpu().numpy(), axis=1)
		ypred_cat.append(tmp_labels)
		label = Variable(sample['label'])
		ytrue_cat.append(sample['label'].squeeze().numpy())
		ypred_bin = np.argmax(output_bin.data.cpu().numpy(), axis=1)
		ypred_res = output_res.data.cpu().numpy()
		ypred_pose.append(kmeans_dict[ypred_bin, :] + ypred_res)
		ytrue_pose.append(sample['ydata'].numpy())
		del xdata, label, output, sample, output_cat, output_bin, output_res
		gc.collect()
	ytrue_cat = np.concatenate(ytrue_cat)
	ypred_cat = np.concatenate(ypred_cat)
	ytrue_pose = np.concatenate(ytrue_pose)
	ypred_pose = np.concatenate(ypred_pose)
	model.train()
	return ytrue_cat, ytrue_pose, ypred_cat, ypred_pose


def save_checkpoint(filename):
	torch.save(model.state_dict(), filename)


# initialization
training_init()
ytrue_cat, ytrue_pose, ypred_cat, ypred_pose = testing()
spio.savemat(results_file, {'ytrue_cat': ytrue_cat, 'ytrue_pose': ytrue_pose, 'ypred_cat': ypred_cat, 'ypred_pose': ypred_pose})
tmp_acc = get_accuracy(ytrue_cat, ypred_cat, num_classes)
tmp_err = get_error2(ytrue_pose, ypred_pose, ytrue_cat, num_classes)
print('Acc: {0} \t Err: {1}'.format(tmp_acc, tmp_err))

for epoch in range(args.num_epochs):
	tic = time.time()
	scheduler.step()
	# training step
	training()
	# save model at end of epoch
	save_checkpoint(model_file)
	# validation
	ytrue_cat, ytrue_pose, ypred_cat, ypred_pose = testing()
	spio.savemat(results_file, {'ytrue_cat': ytrue_cat, 'ytrue_pose': ytrue_pose, 'ypred_cat': ypred_cat, 'ypred_pose': ypred_pose})
	tmp_acc = get_accuracy(ytrue_cat, ypred_cat, num_classes)
	tmp_err = get_error2(ytrue_pose, ypred_pose, ytrue_cat, num_classes)
	print('Acc: {0} \t Err: {1}'.format(tmp_acc, tmp_err))
	# time and output
	toc = time.time() - tic
	print('Epoch: {0} done in time {1}s'.format(epoch, toc))
	# cleanup
	gc.collect()
writer.close()
val_acc = np.stack(val_acc)
val_err = np.stack(val_err)
spio.savemat(plots_file, {'val_acc': val_acc, 'val_err': val_err})

# evaluate the model
ytrue_cat, ytrue_pose, ypred_cat, ypred_pose = testing()
spio.savemat(results_file, {'ytrue_cat': ytrue_cat, 'ytrue_pose': ytrue_pose, 'ypred_cat': ypred_cat, 'ypred_pose': ypred_pose})
tmp_acc = get_accuracy(ytrue_cat, ypred_cat, num_classes)
tmp_err = get_error2(ytrue_pose, ypred_pose, ytrue_cat, num_classes)
print('Acc: {0} \t Err: {1}'.format(tmp_acc, tmp_err))
