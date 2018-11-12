function setupDataFlipped_pascal3d
% function to setup pascal3d+ data with flips. Read the data from PASCAL3D+_release1.1
% stored in db_path. Save resized patch inside ground truth bounding box.
% Usage: setupDataOriginal_pascal3d(cls, db_path, voc_dir);
% cls: class of interest
% db_path: location of the pascal3d+ data. eg: 'D:/datasets/pascal3d/PASCAL3D+_release1.1/';
% voc_dir: location of VOC2012 devkit to get train+val sets. eg: 'D:/datasets/VOCdevkit/VOC2012';

clear; clc;

% paths and variables
db_path = 'data/pascal3d';
voc_dir = 'data/pascal3d/PASCAL/VOCdevkit/VOC2012';
classes = {'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', ...
	'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor'};
num_classes = length(classes);
save_dir = 'data/flipped_new';		% where all this will be stored. change or setup a symbolic link if necessary
anno_dir = fullfile(db_path, 'Annotations');
image_dir = fullfile(db_path, 'Images');
sets_path = fullfile(db_path, 'Image_sets');

% start parallel processing
poolobj = parpool(16);

imagenet_train = cell(1, num_classes);
imagenet_val = cell(1, num_classes);
pascal_train = cell(1, num_classes);
pascal_val = cell(1, num_classes);

for k=1:length(classes)
	cls = classes{k};
	fprintf('cls: %s \n', cls);

	% Imagenet images
	image_path = fullfile(image_dir, sprintf('%s_imagenet', cls));
	anno_path = fullfile(anno_dir, sprintf('%s_imagenet', cls));
	image_extn = '.JPEG';
	% imagenet-train
	image_names = read_file(fullfile(sets_path, sprintf('%s_imagenet_train.txt', cls)));
	ind = zeros(1, length(image_names));
	parfor i = 1:length(image_names)
		ind(i) = process_image(cls, image_names{i}, image_path, anno_path, image_extn, fullfile(save_dir, 'train', cls));
	end
	imagenet_train{k} = image_names(ind>0);
	fprintf('\t imagenet-train: %d / %d \n', sum(ind), length(ind));
	% imagenet-val
	image_names = read_file(fullfile(sets_path, sprintf('%s_imagenet_val.txt', cls)));
	ind = zeros(1, length(image_names));
	parfor i = 1:length(image_names)
		ind(i) = process_image(cls, image_names{i}, image_path, anno_path, image_extn, fullfile(save_dir, 'train', cls));
	end
	imagenet_val{k} = image_names(ind>0);
	fprintf('\t imagenet-val: %d / %d \n', sum(ind), length(ind));

	% PASCAL Images
	image_path = fullfile(image_dir, sprintf('%s_pascal', cls));
	anno_path = fullfile(anno_dir, sprintf('%s_pascal', cls));
	image_extn = '.jpg';
	% pascal-train
	image_names = read_file2(fullfile(voc_dir, 'ImageSets/Main',sprintf('%s_train.txt', cls)));
	ind = zeros(1, length(image_names));
	parfor i = 1:length(image_names)
		ind(i) = process_image(cls, image_names{i}, image_path, anno_path, image_extn, fullfile(save_dir, 'train', cls));
	end
	pascal_train{k} = image_names(ind>0);
	fprintf('\t pascal-train: %d / %d \n', sum(ind), length(ind));
	% pascal-val
	image_names = read_file2(fullfile(voc_dir, 'ImageSets/Main', sprintf('%s_val.txt', cls)));
	ind = zeros(1, length(image_names));
	parfor i = 1:length(image_names)
		ind(i) = process_image2(cls, image_names{i}, image_path, anno_path, image_extn, fullfile(save_dir, 'test', cls));
	end
	pascal_val{k} = image_names(ind>0);
	fprintf('\t pascal-val: %d / %d \n', sum(ind), length(ind));
end

% close parallel threads
delete(poolobj);

% save the train and val data
save(fullfile(save_dir, 'dbinfo'), 'imagenet_train', 'imagenet_val', 'pascal_train', 'pascal_val');


function ind = process_image(cls, image_name, image_path, anno_path, image_extn, save_location)
% function to save the resized extracted patch for future processing
ind = 0;
if ~exist(save_location, 'dir'), mkdir(save_location); end
% load image
img = imread(fullfile(image_path, [image_name, image_extn]));
[nR, nC, d] = size(img);
if(d ~= 3), return; end
% load annotation
tmp = load(fullfile(anno_path, image_name));
objects = tmp.record.objects;
% run through all annotated objects in image
imageid = get_id(image_name);
for j = 1:length(objects)
	object = objects(j);

	% ignore if object not of desired class
	if(~strcmp(object.class, cls)), continue; end
	% ignore if object is truncated or occluded
	if(object.truncated > 0 || object.occluded > 0), continue; end
	% ignore if fine viewpoint has not been annotated
	if(object.viewpoint.distance == 0), continue; end

	% get bbox and pose info
	az = object.viewpoint.azimuth;
	el = object.viewpoint.elevation;
	ct = object.viewpoint.theta;
	d = object.viewpoint.distance;
	bbox = object.bbox;
	if(bbox(1) > nC || bbox(2) > nR), continue; end		% bad bbox
	
	% get patches inside bbox
	patch = get_patch(bbox, img);
	patch_flipped = fliplr(patch);
	
	% save images
	imwrite(patch, fullfile(save_location, sprintf('%s_%sobject%d_a%f_e%f_t%f_d%f.png', cls, imageid, j, az, el, ct, d)));
	imwrite(patch_flipped, fullfile(save_location, sprintf('%s_%sobject%d_a%f_e%f_t%f_d%f.png', cls, imageid, j, -az, el, -ct, d)));
	ind = 1;
end


function patch = get_patch(bbox, img)
% function to get patch inside gt-bbox
[nR, nC, ~] = size(img);
% extract patch inside bounding box
x1 = max(1, round(bbox(1))); x2 = min(nC, round(bbox(3)));
y1 = max(1, round(bbox(2))); y2 = min(nR, round(bbox(4)));
patch = img(y1:y2, x1:x2, :);
% resize if necessary
scale = [size(patch, 1)/224, size(patch, 2)/224];
if any(scale>1), patch = imresize(patch, 1/max(scale)); end


function image_list = read_file(filename)
% function to read file in filename and retrieve text info stored in it
disp(filename);
fid=fopen(filename,'r');
tmp=textscan(fid,'%s');
image_list=tmp{1};
fclose(fid);


function image_list=read_file2(filename)
% function to read pascal test-train splits
disp(filename);
fid=fopen(filename,'r');
tmp=textscan(fid,'%s %d');
image_list=tmp{1};
is_class=tmp{2};
image_list=image_list(is_class>0);


function ind = process_image2(cls, image_name, image_path, anno_path, image_extn, save_location)
% function to save the resized extracted patch for future processing
ind = 0;
if ~exist(save_location, 'dir'), mkdir(save_location); end
% load image
img = imread(fullfile(image_path, [image_name, image_extn]));
[nR, nC, d] = size(img);
if(d ~= 3), return; end
% load annotation
tmp = load(fullfile(anno_path, image_name));
objects = tmp.record.objects;
% run through all annotated objects in image
imageid = get_id(image_name);
for j = 1:length(objects)
	object = objects(j);

	% ignore if object not of desired class
	if(~strcmp(object.class, cls)), continue; end
	% ignore if object is truncated or occluded
	if(object.truncated > 0 || object.occluded > 0), continue; end
	% ignore if fine viewpoint has not been annotated
	if(object.viewpoint.distance == 0), continue; end

	% get bbox and pose info
	az = object.viewpoint.azimuth;
	el = object.viewpoint.elevation;
	ct = object.viewpoint.theta;
	d = object.viewpoint.distance;
	bbox = object.bbox;
	if(bbox(1) > nC || bbox(2) > nR), continue; end		% bad bbox
	
	% get patch inside bbox without any augmentation
	patch = get_patch(bbox, img);
	imwrite(patch, fullfile(save_location, sprintf('%s_%sobject%d_a%f_e%f_t%f_d%f.png', cls, imageid, j, az, el, ct, d)));
	ind = 1;
end

function imageid = get_id(image_name)
ind = strfind(image_name, '_');
imageid = image_name(setdiff(1:length(image_name), ind));
