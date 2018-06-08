# -*- coding: utf-8 -*-
"""
Feature models of interest
"""

from torch import nn
from torchvision import models


# feature model based on resnet architecture
class resnet_model(nn.Module):

	def __init__(self, model_type='resnet50', layer_type='layer4'):
		super().__init__()
		# get model
		if model_type == 'resnet50':
			original_model = models.resnet50(pretrained=True)
		elif model_type == 'resnet101':
			original_model = models.resnet101(pretrained=True)
		else:
			raise NameError('Unknown model_type passed')
		# get requisite layer
		if layer_type == 'layer2':
			num_layers = 6
			pool_size = 28
		elif layer_type == 'layer3':
			num_layers = 7
			pool_size = 14
		elif layer_type == 'layer4':
			num_layers = 8
			pool_size = 7
		else:
			raise NameError('Uknown layer_type passed')
		self.features = nn.Sequential(*list(original_model.children())[:num_layers])
		self.avgpool = nn.AvgPool2d(pool_size, stride=1)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		return x


class vgg_model(nn.Module):

	def __init__(self, model_type='vgg13', layer_type='fc6'):
		super().__init__()
		# get model
		if model_type == 'vgg13':
			self.original_model = models.vgg13_bn(pretrained=True)
		elif model_type == 'vgg16':
			self.original_model = models.vgg16_bn(pretrained=True)
		else:
			raise NameError('Unknown model_type passed')
		self.features = self.original_model.features
		if layer_type == 'fc6':
			self.classifier = nn.Sequential(*list(self.original_model.classifier.children())[:2])
		elif layer_type == 'fc7':
			self.classifier = nn.Sequential(*list(self.original_model.classifier.children())[:-2])
		else:
			raise NameError('Uknown layer_type passed')

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x
