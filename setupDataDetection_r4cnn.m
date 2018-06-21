function setupDataDetection_r4cnn
% function that reads the bboxes provided by Render-for-CNN to evaluate my
% models. Extract images using bboxes provided
clear; clc; close all;

% relevant paths
pascal3d_path = 'data/pascal3d';
db_path = fullfile(pascal3d_path, 'PASCAL/VOCdevkit/VOC2012');
mat_path = 'data/r4cnn_dets';
img_path = fullfile(db_path, 'JPEGImages');
sets_file = fullfile(db_path, 'ImageSets/Main');
dest_path = fullfile(mat_path, 'all');
patch_size = [224, 224];

% get list of all test images
fid = fopen(fullfile(sets_file, 'val.txt'), 'r');
tmp = textscan(fid, '%s');
image_names = tmp{1};
fclose(fid);

classes = {'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', ...
	'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor'};
num_classes = 12;
% load all detections
dets = cell(1, num_classes);
for i = 1:num_classes
	tmp = load(fullfile(mat_path, sprintf('%s_pruned_boxes_voc_2012_val_bbox_reg', classes{i})));
	dets{i} = tmp.boxes;	
end

for i = 1:length(image_names)
	image_name = image_names{i};
	bboxes = cell(1, num_classes); 
	labels = cell(1, num_classes);
	for j = 1:num_classes
		bboxes{j} = dets{j}{i};
		labels{j} = i*ones(size(dets{j}{i}, 1), 1);
	end
	bboxes = cat(1, bboxes{:});
	labels = cat(1, labels{:});
	xdata = cell(1, size(bboxes, 1));
	% read image
	img = imread(fullfile(img_path, sprintf('%s.jpg', image_name)));
	for k = 1:size(bboxes, 1)
		patch = get_patch(bboxes(k, :), img, patch_size);
		xdata{k} = shiftdim(patch, -1); 
	end
	xdata = cat(1, xdata{:});
	fprintf('image: %d \t num_boxes: %d \n', i, length(labels));
	save(fullfile(dest_path, image_name), 'xdata', 'bboxes', 'labels', '-v7.3');
end

function patch = get_patch(bbox, img, patch_size)
% function to get patch inside gt-bbox
[nR, nC, ~] = size(img);
% extract patch inside bounding box
x1 = max(1, round(bbox(1))); x2 = min(nC, round(bbox(3)));
y1 = max(1, round(bbox(2))); y2 = min(nR, round(bbox(4)));
patch = img(y1:y2, x1:x2, :);
% resize patch to canonical size
patch = imresize(patch, patch_size) ;
