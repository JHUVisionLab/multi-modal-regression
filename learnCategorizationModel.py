import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from helperFunctions import classes, get_accuracy
from dataGenerators import ImagesAll, TestImages, my_collate
from featureModels import resnet_model

import numpy as np
import scipy.io as spio
import gc
import os
import progressbar
import time
import sys

if len(sys.argv) > 1:
	os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]


# relevant paths
train_path = 'data/flipped_new/train/'
test_path = 'data/flipped_new/test/'

# save things here
save_str = 'category_all_10'
results_file = os.path.join('results', save_str)
model_file = os.path.join('models', save_str + '.tar')
plots_file = os.path.join('plots', save_str)

# relevant variables
num_workers = 8
num_classes = len(classes)
init_lr = 0.0001
num_epochs = 50
N0 = 2048
batch_size = 8

# datasets
train_data = ImagesAll(train_path, 'real')
test_data = TestImages(test_path)
# setup data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=32)
print('Train: {0} \t Test: {1}'.format(len(train_loader), len(test_loader)))


# MODEL
# my model for pose estimation: feature model + 1layer pose model x 12
class my_model(nn.Module):
	def __init__(self):
		super().__init__()
		self.feature_model = resnet_model('resnet50', 'layer4').cuda()
		self.fc = nn.Linear(N0, num_classes).cuda()

	def forward(self, x):
		x = self.feature_model(x)
		x = self.fc(x)
		return x


model = my_model()
for param in model.feature_model.parameters():
	param.requires_grad = False
model.eval()
# print(model)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda ep: 1./(1. + ep))
criterion = nn.CrossEntropyLoss().cuda()


# OPTIMIZATION functions
def training():
	# model.train()
	bar = progressbar.ProgressBar(max_value=len(train_loader))
	for i, sample in enumerate(train_loader):
		# forward steps
		xdata = Variable(sample['xdata'].cuda())
		ydata = Variable(sample['label'].cuda()).squeeze()
		output = model(xdata)
		loss = criterion(output, ydata)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# store
		bar.update(i)
		# cleanup
		del xdata, ydata, output, loss, sample
		gc.collect()
	train_loader.dataset.shuffle_images()


def testing():
	# model.eval()
	ypred = []
	ytrue = []
	for i, sample in enumerate(test_loader):
		xdata = Variable(sample['xdata'].cuda())
		output = model(xdata)
		tmp_labels = torch.argmax(output, dim=1)
		ypred.append(tmp_labels.data.cpu().numpy())
		ytrue.append(sample['label'].squeeze().numpy())
		del xdata, output, sample, tmp_labels
		gc.collect()
	ypred = np.concatenate(ypred)
	ytrue = np.concatenate(ytrue)
	# model.train()
	return ytrue, ypred


def save_checkpoint(filename):
	torch.save(model.state_dict(), filename)


for epoch in range(num_epochs):
	tic = time.time()
	scheduler.step()
	# training step
	training()
	# save model at end of epoch
	save_checkpoint(model_file)
	# evaluate
	ygt, ypred = testing()
	print('Acc: {0}'.format(get_accuracy(ygt, ypred, num_classes)))
	spio.savemat(results_file, {'ygt': ygt, 'ypred': ypred})
	# time and output
	toc = time.time() - tic
	print('Epoch: {0} in time {1}s'.format(epoch, toc))
	# cleanup
	gc.collect()

# evaluate the model
ygt, ypred = testing()
print('Acc: {0}'.format(get_accuracy(ygt, ypred, num_classes)))
spio.savemat(results_file, {'ygt': ygt, 'ypred': ypred})
