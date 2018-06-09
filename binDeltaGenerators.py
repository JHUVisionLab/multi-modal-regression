from dataGenerators import ImagesAll
import torch
import numpy as np
import pickle
from scipy.spatial.distance import cdist
from quaternion import convert_dictionary


class GBDGenerator(ImagesAll):
	def __init__(self, db_path, db_type, kmeans_file):
		# initialize the renderedImages dataset first
		super().__init__(db_path, db_type)
		# add the kmeans part
		self.kmeans = pickle.load(open(kmeans_file, 'rb'))
		self.num_clusters = self.kmeans.n_clusters

	def __len__(self):
		return np.amax(self.num_images)

	def __getitem__(self, idx):
		# run the item handler of the renderedImages dataset
		sample = super().__getitem__(idx)
		# update the ydata target using kmeans dictionary
		ydata = sample['ydata'].numpy()
		# bin part
		ydata_bin = self.kmeans.predict(ydata)
		sample['ydata_bin'] = torch.from_numpy(ydata_bin).long()
		# residual part
		ydata_res = ydata - self.kmeans.cluster_centers_[ydata_bin, :]
		sample['ydata_res'] = torch.from_numpy(ydata_res).float()
		return sample


class XPBDGenerator(ImagesAll):
	def __init__(self, db_path, db_type, kmeans_file):
		# initialize the renderedImages dataset first
		super().__init__(db_path, db_type)
		# add the kmeans part
		self.kmeans = pickle.load(open(kmeans_file, 'rb'))
		self.num_clusters = self.kmeans.n_clusters

	def __len__(self):
		return np.amax(self.num_images)

	def __getitem__(self, idx):
		# run the item handler of the renderedImages dataset
		sample = super().__getitem__(idx)
		# update the ydata target using kmeans dictionary
		ydata = sample['ydata'].numpy()
		# bin part
		ydata_bin = np.exp(-10.0*cdist(ydata, self.kmeans.cluster_centers_, 'sqeuclidean'))
		ydata_bin = ydata_bin/np.sum(ydata_bin, axis=1, keepdims=True)
		sample['ydata_bin'] = torch.from_numpy(ydata_bin).float()
		# residual part
		ydata_res = ydata - np.dot(ydata_bin, self.kmeans.cluster_centers_)     # need to think more about m4
		sample['ydata_res'] = torch.from_numpy(ydata_res).float()
		return sample


class GBDGeneratorQ(ImagesAll):
	def __init__(self, db_path, db_type, kmeans_file):
		# initialize the renderedImages dataset first
		super().__init__(db_path, db_type, 'quaternion')
		# add the kmeans part
		self.kmeans = pickle.load(open(kmeans_file, 'rb'))
		self.num_clusters = self.kmeans.n_clusters
		self.kmeans.cluster_centers_ = convert_dictionary(self.kmeans.cluster_centers_)

	def __len__(self):
		return np.amax(self.num_images)

	def __getitem__(self, idx):
		# run the item handler of the renderedImages dataset
		sample = super().__getitem__(idx)
		# update the ydata target using kmeans dictionary
		ydata = sample['ydata'].numpy()
		# bin part
		ydata_bin = self.kmeans.predict(ydata)
		sample['ydata_bin'] = torch.from_numpy(ydata_bin).long()
		# residual part
		ydata_res = ydata - self.kmeans.cluster_centers_[ydata_bin, :]
		sample['ydata_res'] = torch.from_numpy(ydata_res).float()
		return sample


