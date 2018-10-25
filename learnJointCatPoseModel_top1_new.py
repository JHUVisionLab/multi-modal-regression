# -*- coding: utf-8 -*-
"""
Joint Cat & Pose model (Top1) with Geodesic Bin and Delta model for the axis-angle representation
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataGenerators import TestImages, my_collate
from binDeltaGenerators import GBDGenerator
from binDeltaModels import OneBinDeltaModel, OneDeltaPerBinModel
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
parser.add_argument('--save_str', type=str)
parser.add_argument('--dict_size', type=int, default=200)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--feature_network', type=str, default='resnet')
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--multires', type=bool, default=False)
parser.add_argument('--db_type', type=str, default='clean')
parser.add_argument('--init_lr', type=float, default=1e-4)
parser.add_argument('--alpha', type=float, default=1)
args = parser.parse_args()
print(args)
# assign GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# save stuff here
init_model_file = os.path.join('models', args.save_str + '_cat.tar')
model_file = os.path.join('models', args.save_str + '_top1_new.tar')
results_file = os.path.join('results', args.save_str + '_top1_new_' + args.db_type)
plots_file = os.path.join('plots', args.save_str + '_top1_new_' + args.db_type)
log_dir = os.path.join('logs', args.save_str + '_top1_new_' + args.db_type)

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
real_path = os.path.join(db_path, 'train')
render_path = 'data/renderforcnn'
test_path = os.path.join(db_path, 'test')

# loss
ce_loss = nn.CrossEntropyLoss().cuda()
gve_loss = geodesic_loss().cuda()

# DATA
# datasets
real_data = GBDGenerator(real_path, 'real', kmeans_file)
render_data = GBDGenerator(render_path, 'render', kmeans_file)
test_data = TestImages(test_path)
# setup data loaders
real_loader = DataLoader(real_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
render_loader = DataLoader(render_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
test_loader = DataLoader(test_data, batch_size=32)
print('Real: {0} \t Render: {1} \t Test: {2}'.format(len(real_loader), len(render_loader), len(test_loader)))
max_iterations = min(len(real_loader), len(render_loader))

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
		Pc = F.softmax(y0, dim=1)
		y1 = torch.stack([self.bin_models[i](x) for i in range(self.num_classes)]).permute(1, 2, 0)
		Pl = F.softmax(y1, dim=1)
		Plc = Pl * torch.unsqueeze(Pc, dim=1)
		ind = torch.argmax(Plc.view(x.size(0), -1), dim=1, keepdim=True)
		ip = ind/self.num_classes
		ic = torch.fmod(ind, self.num_classes)
		label = torch.zeros(ic.size(0), self.num_classes).scatter_(1, ic.data.cpu(), 1.0)
		label = Variable(label.unsqueeze(2).cuda())
		y1 = torch.squeeze(torch.bmm(y1, label), 2)
		if not args.multires:
			y2 = torch.stack([self.res_models[i](x) for i in range(self.num_classes)]).permute(1, 2, 0)
			y2 = torch.squeeze(torch.bmm(y2, label), 2)
		else:
			y2 = torch.stack([self.res_models[i](x) for i in range(self.num_classes * self.num_clusters)])
			y2 = y2.view(self.num_classes, self.num_clusters, -1, self.ndim).permute(1, 2, 3, 0)
			y2 = torch.squeeze(torch.matmul(y2, label), 3)
			pose_label = torch.zeros(ip.size(0), self.num_clusters).scatter_(1, ip.data.cpu(), 1.0)
			pose_label = Variable(pose_label.unsqueeze(2).cuda())
			y2 = torch.squeeze(torch.bmm(y2.permute(1, 2, 0), pose_label), 2)
		return [y0, y1, y2, Plc]   # cat, pose_bin, pose_delta


model = JointCatPoseModel(orig_model)
model.load_state_dict(torch.load(init_model_file))
# print(model)


def my_schedule(ep):
	return 1. / (1. + ep)
	# return 10**-(ep//10)/(1 + ep % 10)


optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, my_schedule)
writer = SummaryWriter(log_dir)
count = 0
val_err = []
val_acc = []


def training():
	global count, val_acc, val_err
	model.train()
	bar = progressbar.ProgressBar(max_value=max_iterations)
	for i, (sample_real, sample_render) in enumerate(zip(real_loader, render_loader)):
		# forward steps
		# output
		label_real = Variable(sample_real['label'].squeeze().cuda())
		ydata_bin_real = Variable(sample_real['ydata_bin'].cuda())
		ydata_real = Variable(sample_real['ydata'].cuda())
		xdata_real = Variable(sample_real['xdata'].cuda())
		output_real = model(xdata_real)
		output_cat_real = output_real[0]
		output_bin_real = output_real[1]
		output_res_real = output_real[2]
		label_render = Variable(sample_render['label'].squeeze().cuda())
		ydata_bin_render = Variable(sample_render['ydata_bin'].cuda())
		ydata_render = Variable(sample_render['ydata'].cuda())
		xdata_render = Variable(sample_render['xdata'].cuda())
		output_render = model(xdata_render)
		output_cat_render = output_render[0]
		output_bin_render = output_render[1]
		output_res_render = output_render[2]
		output_bin = torch.cat((output_bin_real, output_bin_render))
		output_res = torch.cat((output_res_real, output_res_render))
		ydata_bin = torch.cat((ydata_bin_real, ydata_bin_render))
		ydata = torch.cat((ydata_real, ydata_render))
		# loss
		Lc_cat = ce_loss(output_cat_real, label_real)   # use only real images for category loss
		Lc_pose = ce_loss(output_bin, ydata_bin)        # use all images for pose loss - bin part
		ind = torch.argmax(output_bin, dim=1)
		y = torch.index_select(cluster_centers_, 0, ind) + output_res
		Lr = gve_loss(y, ydata)                         # gve loss on final pose
		loss = 0.1*Lc_cat + Lc_pose + args.alpha*Lr
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
		del label_real, ydata_bin_real, ydata_real, xdata_real, output_real, output_res_real, output_bin_real, output_cat_real
		del label_render, ydata_bin_render, ydata_render, xdata_render, output_render, output_res_render, output_bin_render, output_cat_render
		del	output_bin, output_res, ydata_bin, ydata, Lc_cat, Lc_pose, Lr, loss
		bar.update(i+1)
	real_loader.dataset.shuffle_images()
	render_loader.dataset.shuffle_images()


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
		joint_probs = output[3]
		ind = torch.argmax(joint_probs.view(xdata.size(0), -1), dim=1)
		ip = ind/num_classes
		ic = torch.fmod(ind, num_classes)
		tmp_labels = ic.data.cpu().numpy()
		ypred_cat.append(tmp_labels)
		label = Variable(sample['label'])
		ytrue_cat.append(sample['label'].squeeze().numpy())
		ypred_bin = ip.data.cpu().numpy()
		ypred_res = output_res.data.cpu().numpy()
		ypred_pose.append(kmeans_dict[ypred_bin, :] + ypred_res)
		ytrue_pose.append(sample['ydata'].numpy())
		del xdata, label, output, sample, output_cat, output_bin, output_res, joint_probs, ind, ip, ic
		gc.collect()
	ytrue_cat = np.concatenate(ytrue_cat)
	ypred_cat = np.concatenate(ypred_cat)
	ytrue_pose = np.concatenate(ytrue_pose)
	ypred_pose = np.concatenate(ypred_pose)
	model.train()
	return ytrue_cat, ytrue_pose, ypred_cat, ypred_pose


def save_checkpoint(filename):
	torch.save(model.state_dict(), filename)


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
	writer.add_scalar('val_acc', tmp_acc, count)
	writer.add_scalar('val_err', tmp_err, count)
	val_acc.append(tmp_acc)
	val_err.append(tmp_err)
	# time and output
	toc = time.time() - tic
	print('Epoch: {0} done in time {1}s'.format(epoch, toc))
	# cleanup
	gc.collect()
writer.close()
val_acc = np.stack(val_acc)
val_err = np.stack(val_err)
spio.savemat(plots_file, {'val_acc': val_acc, 'val_err': val_err})
