# -*- coding: utf-8 -*-
"""
Geodesic Bin and Delta model for the axis-angle representation
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataGenerators import Pascal3dAll, my_collate
from binDeltaGenerators import XPBDGenerator
from axisAngle import get_error2, geodesic_loss
from binDeltaModels import OneBinDeltaModel, ProbabilisticOneDeltaPerBinModel
from binDeltaLosses import RelaXedProbabilisticLoss, RelaXedProbabilisticMultiresLoss
from helperFunctions import classes

import numpy as np
import scipy.io as spio
import math
import gc
import os
import time
import progressbar
import pickle
import argparse
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='RelaXed Probabilistic Bin & Delta Model')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--render_path', type=str, default='data/renderforcnn/')
parser.add_argument('--augmented_path', type=str, default='data/augmented2/')
parser.add_argument('--pascal3d_path', type=str, default='data/original')
parser.add_argument('--save_str', type=str)
parser.add_argument('--dict_size', type=int, default=200)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--feature_network', type=str, default='resnet')
parser.add_argument('--N0', type=int, default=2048)
parser.add_argument('--N1', type=int, default=1000)
parser.add_argument('--N2', type=int, default=500)
parser.add_argument('--N3', type=int, default=100)
parser.add_argument('--init_lr', type=float, default=1e-4)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--max_iterations', type=float, default=np.inf)
parser.add_argument('--multires', type=bool, default=False)
parser.add_argument('--alpha', type=float, default=1.0)
args = parser.parse_args()
print(args)
# assign GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# save stuff here
results_file = os.path.join('results', args.save_str)
model_file = os.path.join('models', args.save_str + '.tar')
plots_file = os.path.join('plots', args.save_str)
log_dir = os.path.join('logs', args.save_str)

# gmm data
gmm_file = 'data/gmm_dictionary_axis_angle_' + str(args.dict_size) + '.pkl'
gmm = pickle.load(open(gmm_file, 'rb'))
gmm_dict = gmm.means_
num_clusters = gmm.n_components
cluster_centers = Variable(torch.from_numpy(gmm_dict).float()).cuda()

# relevant variables
ndim = 3
num_classes = len(classes)

# loss
kl_div = nn.KLDivLoss().cuda()
gve_loss = geodesic_loss(reduce=False).cuda()

# DATA
# datasets
real_data = XPBDGenerator(args.augmented_path, 'real', gmm_file)
render_data = XPBDGenerator(args.render_path, 'render', gmm_file)
test_data = Pascal3dAll(args.pascal3d_path, 'test')
# setup data loaders
real_loader = DataLoader(real_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
render_loader = DataLoader(render_data, batch_size=args.num_workers, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=my_collate)
test_loader = DataLoader(test_data, batch_size=32, collate_fn=my_collate)
print('Real: {0} \t Render: {1} \t Test: {2}'.format(len(real_loader), len(render_loader), len(test_loader)))

if np.isinf(args.max_iterations):
	max_iterations = min(len(real_loader), len(render_loader))
else:
	max_iterations = args.max_iterations

# my_model
if not args.multires:
	model = OneBinDeltaModel(args.feature_network, num_classes, num_clusters, args.N0, args.N1, args.N2, ndim)
else:
	model = ProbabilisticOneDeltaPerBinModel(args.feature_network, num_classes, num_clusters, args.N0, args.N1, args.N2, args.N3, ndim)

# print(model)
# loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
# store stuff
writer = SummaryWriter(log_dir)
count = 0
val_loss = []
s = 0


# OPTIMIZATION functions
def training():
	global count, val_loss, s
	model.train()
	bar = progressbar.ProgressBar(max_value=max_iterations)
	for i, (sample_real, sample_render) in enumerate(zip(real_loader, render_loader)):
		# forward steps
		xdata_real = Variable(sample_real['xdata'].cuda())
		label_real = Variable(sample_real['label'].cuda())
		output_real = model(xdata_real, label_real)
		xdata_render = Variable(sample_render['xdata'].cuda())
		label_render = Variable(sample_render['label'].cuda())
		output_render = model(xdata_render, label_render)
		# loss
		ydata_bin = torch.cat((Variable(sample_real['ydata_bin'].cuda()), Variable(sample_render['ydata_bin'].cuda())))
		output_bin = torch.cat((output_real[0], output_render[0]))
		Lc = kl_div(F.log_softmax(output_bin, dim=1), ydata_bin)
		ydata = torch.cat((Variable(sample_real['ydata'].cuda()), Variable(sample_render['ydata'].cuda())))
		output_res = torch.cat((output_real[1], output_render[1]))
		if not args.multires:
			Lr = torch.stack([gve_loss(ydata, torch.add(output_res, 1.0, cluster_centers.index_select(0, Variable(j*torch.ones(1).long().cuda())))) for j in range(num_clusters)])
		else:
			y = cluster_centers + output_res
			Lr = torch.stack([gve_loss(ydata, torch.squeeze(y.index_select(1, Variable(j * torch.ones(1).long().cuda())))) for j in range(num_clusters)])
		Lr = torch.mean(torch.sum(torch.mul(F.softmax(output_bin, dim=1), torch.t(Lr)), dim=1))
		loss = Lc + math.exp(-s)*Lr + s
		# updates
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		s = math.log(Lr)
		# store
		count += 1
		writer.add_scalar('train_loss', loss.item(), count)
		writer.add_scalar('alpha', math.exp(-s), count)
		if i % 1000 == 0:
			ytest, yhat_test, test_labels = testing()
			spio.savemat(results_file, {'ytest': ytest, 'yhat_test': yhat_test, 'test_labels': test_labels})
			tmp_val_loss = get_error2(ytest, yhat_test, test_labels, num_classes)
			writer.add_scalar('val_loss', tmp_val_loss, count)
			val_loss.append(tmp_val_loss)
		# cleanup
		del xdata_real, xdata_render, label_real, label_render
		del ydata_bin, output_bin, ydata, output_res
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
		if not args.multires:
			# ypred_bin = F.softmax(output[0], dim=1).data.cpu().numpy()
			# ypred_res = output[1].data.cpu().numpy()
			# tmp_ypred = np.dot(ypred_bin, gmm_dict) + ypred_res
			ypred_bin = np.argmax(output[0].data.cpu().numpy(), axis=1)
			ypred_res = output[1].data.cpu().numpy()
			tmp_ypred = gmm_dict[ypred_bin, :] + ypred_res
		else:
			# ypred_bin = F.softmax(output[0], dim=1).data.cpu().numpy()
			# ypred_res = output[1].data.cpu().numpy()
			# tmp_ypred = np.stack([np.dot(gmm_dict.T + ypred_res[i].T, ypred_bin[i]) for i in range(ypred_bin.shape[0])])
			ypred_bin = np.argmax(output[0].data.cpu().numpy(), axis=1)
			ypred_res = output[1].data.cpu().numpy()
			tmp_ypred = np.stack(
				[gmm_dict[ypred_bin[i]] + ypred_res[i, ypred_bin[i], :] for i in range(ypred_bin.shape[0])])
		ypred.append(tmp_ypred)
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


# evaluate the model
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
