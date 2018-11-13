function setupDataFlipped_objectnet3d
% function to setup objectnet3d data with flips. 

clear; clc;

% paths and variables
db_path = '/cis/data/msid_io4/objectnet3d';
save_dir = 'data/objectnet';		% where all this will be stored. change or setup a symbolic link if necessary
anno_dir = fullfile(db_path, 'Annotations');
image_dir = fullfile(db_path, 'Images');
sets_path = fullfile(db_path, 'Image_sets');

% get category info
cls_file = fullfile(db_path, 'Image_sets/classes.txt');
fid = fopen(cls_file, 'r');
tmp = textscan(fid, '%s');
fclose(fid);
classes = tmp{1};
num_classes = length(classes);

% get trainval images
fid = fopen(fullfile(sets_path, 'trainval.txt'), 'r');
tmp = textscan(fid, '%s');
fclose(fid);
trainval_images = tmp{1};

% get test images
fid = fopen(fullfile(sets_path, 'test.txt'), 'r');
tmp = textscan(fid, '%s');
fclose(fid);
test_images = tmp{1};

% setup folders to store data
train_path = fullfile(save_dir, 'train'); mkdir(train_path);
test_path = fullfile(save_dir, 'test'); mkdir(test_path);
for i = 1:num_classes
	mkdir(fullfile(train_path, classes{i}));
	mkdir(fullfile(test_path, classes{i}));
end

% start parallel processing
poolobj = parpool(16);

parfor k = 1:length(trainval_images)
	fprintf('Train \t k: %d \n', k);
	process_train_image(trainval_images{k}, image_dir, anno_dir, train_path);
end

parfor k = 1:length(test_images)
	fprintf('Test \t k: %d \n', k);
	process_test_image(test_images{k}, image_dir, anno_dir, test_path);
end

% close parallel threads
delete(poolobj);

% store dataset info
save(fullfile(save_dir, 'dbinfo'), 'classes', 'trainval_images', 'test_images');
% store test-train image info per category
for i = 1:num_classes
	cls = classes{i};
	% train
	cls_train_path = fullfile(train_path, cls);
	files = dir(fullfile(cls_train_path, '*.png'));
	file_names = {files(:).name};
	image_names = cellfun(@(x) x(1:end-4), file_names, 'uniformoutput', false);
	save(fullfile(train_path, sprintf('%s_info', cls)), 'image_names');
	fprintf('Found %5d training images for Cls: %s \n', length(image_names), cls);
	% test
	cls_test_path = fullfile(test_path, cls);
	files = dir(fullfile(cls_test_path, '*.png'));
	file_names = {files(:).name};
	image_names = cellfun(@(x) x(1:end-4), file_names, 'uniformoutput', false);
	save(fullfile(test_path, sprintf('%s_info', cls)), 'image_names');
	fprintf('Found %5d testing images for Cls: %s \n', length(image_names), cls);
end

function process_train_image(image_name, image_dir, anno_dir, train_path)
imageid = get_id(image_name);
% read image
img = imread(fullfile(image_dir, [image_name, '.JPEG']));
% read annotation
tmp = load(fullfile(anno_dir, image_name));
objects = tmp.record.objects;
for i = 1:length(objects)
	% get object info
	object = objects(i);
	cls = object.class;
	clsid = get_id(cls);
	bbox = object.bbox;
	% get viewpoint info
	viewpoint = object.viewpoint;
	if ~isfield(viewpoint, 'azimuth') || isempty(viewpoint.azimuth) 
		az = viewpoint.azimuth_coarse; 
	else 
		az = viewpoint.azimuth; 
	end
	if ~isfield(viewpoint, 'elevation') || isempty(viewpoint.elevation) 
		el = viewpoint.elevation_coarse; 
	else 
		el = viewpoint.elevation; 
	end
	ct = viewpoint.theta;
	d = viewpoint.distance;
	patch = get_patch(bbox, img);
	patch_flipped = fliplr(patch);
	% save images
	save_location = fullfile(train_path, cls);
	imwrite(patch, fullfile(save_location, sprintf('%s_%sobject%d_a%f_e%f_t%f_d%f.png', clsid, imageid, i, az, el, ct, d)));
	imwrite(patch_flipped, fullfile(save_location, sprintf('%s_%sobject%d_a%f_e%f_t%f_d%f.png', clsid, imageid, i, -az, el, -ct, d)));
end

function process_test_image(image_name, image_dir, anno_dir, test_path)
imageid = get_id(image_name);
% read image
img = imread(fullfile(image_dir, [image_name, '.JPEG']));
% read annotation
tmp = load(fullfile(anno_dir, image_name));
objects = tmp.record.objects;
for i = 1:length(objects)
	% get object info
	object = objects(i);
	cls = object.class;
	clsid = get_id(cls);
	bbox = object.bbox;
	% get viewpoint info
	viewpoint = object.viewpoint;
	if ~isfield(viewpoint, 'azimuth') || isempty(viewpoint.azimuth) 
		az = viewpoint.azimuth_coarse; 
	else 
		az = viewpoint.azimuth; 
	end
	if ~isfield(viewpoint, 'elevation') || isempty(viewpoint.elevation) 
		el = viewpoint.elevation_coarse; 
	else 
		el = viewpoint.elevation; 
	end
	ct = viewpoint.theta;
	d = viewpoint.distance;
	patch = get_patch(bbox, img);
	% save images
	save_location = fullfile(test_path, cls);
	imwrite(patch, fullfile(save_location, sprintf('%s_%sobject%d_a%f_e%f_t%f_d%f.png', clsid, imageid, i, az, el, ct, d)));
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

function imageid = get_id(image_name)
ind = strfind(image_name, '_');
imageid = image_name(setdiff(1:length(image_name), ind));
