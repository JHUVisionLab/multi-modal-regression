function computeAVP(filename, nbins, dets_path)
% function to compute the detection metrics

% divide the azimuth space into bins
azimuth_interval = [0 (360/(nbins*2)):(360/nbins):360-(360/(nbins*2))];

% relevant paths
pascal3d_path = 'data/pascal3d/';
anno_path = fullfile(pascal3d_path, 'Annotations');
% dets_path = 'data/r4cnn_dets/';

classes = {'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', ...
	'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor'};
num_classes = length(classes);

% load list of all images
tmp = load(fullfile(dets_path, 'dbinfo'));
image_names = tmp.image_names;
N = length(image_names);

% load predictions
tmp = load(filename);
boxes_all = tmp.bbox;
ypred_all = tmp.ypred;
labels_all = tmp.labels;

for cls_id = 1:num_classes
	cls = classes{cls_id};
	energy = [];
	correct = [];
	correct_view = [];
	overlap = [];
	count = zeros(N, 1);
	num = zeros(N, 1);
	num_pr = 0;
	err = [];
	for i = 1:N
		% fprintf('cls: %s \t i: %d/%d \n', cls, i, N);
		image_name = image_names{i};
		% check if this annotation exists
		filename = fullfile(anno_path, sprintf('%s_pascal', cls), [image_name, '.mat']);
		if ~exist(filename, 'file'), continue; end
		% load annotation
		tmp = load(filename);
		record = tmp.record;
		objects = record.objects;
		% check cls
		clsinds = strcmp(cls, {objects(:).class});
		diff = [objects(:).difficult];
		clsinds = find(clsinds & ~diff);
		% get gt-bbox and gt-pose
		n = numel(clsinds);
		bbox = zeros(n, 4); view_az = zeros(n, 1); az_gt = zeros(n, 1);
		for j = 1:n
			bbox(j, :) = objects(clsinds(j)).bbox;
			viewpoint = objects(clsinds(j)).viewpoint;
			if viewpoint.distance == 0
				az = viewpoint.azimuth_coarse;
			else
				az = viewpoint.azimuth;
			end
			view_az(j) = az;
			az_gt(j) = find_interval(az, azimuth_interval);
		end
		count(i) = n;
		det = zeros(n, 1);
		
		% predicted bbox
		ind = find(labels_all{i} == (cls_id-1));
		num(i) = length(ind);
		dets = boxes_all{i}(ind, :);
		ypred = ypred_all{i}(ind, :);
		for j = 1:num(i)
			num_pr = num_pr + 1;
			energy(num_pr) = dets(j, 5);
			bbox_pr = dets(j, 1:4);
			az_pred = get_azimuth(ypred(j, :));
			az_pr = find_interval(az_pred, azimuth_interval);
			
			if ~isempty(bbox)
				o = box_overlap(bbox, bbox_pr);
				[maxo, index] = max(o);
				if maxo >= 0.5 && det(index)==0
					overlap{num_pr} = index;
					correct(num_pr) = 1;
					det(index) = 1;
					err = [err, abs(az_pred - view_az(index))];
					if az_pr == az_gt(index)
						correct_view(num_pr) = 1;
					else
						correct_view(num_pr) = 0;
					end
				else
					overlap{num_pr} = [];
					correct(num_pr) = 0;
					correct_view(num_pr) = 0;
				end
			else
				overlap{num_pr} = [];
				correct(num_pr) = 0;
				correct_view(num_pr) = 0;
			end
		end
	end
	overlap = overlap';

	[threshold, index] = sort(energy, 'descend');
	correct = correct(index);
	correct_view = correct_view(index);
	n = numel(threshold);
	recall = zeros(n,1);
	precision = zeros(n,1);
	accuracy = zeros(n,1);
	num_correct = 0;
	num_correct_view = 0;
	for i = 1:n
		% compute precision
		num_positive = i;
		num_correct = num_correct + correct(i);
		if num_positive ~= 0
			precision(i) = num_correct / num_positive;
		else
			precision(i) = 0;
		end

		% compute accuracy
		num_correct_view = num_correct_view + correct_view(i);
		if num_correct ~= 0
			accuracy(i) = num_correct_view / num_positive;
		else
			accuracy(i) = 0;
		end

		% compute recall
		recall(i) = num_correct / sum(count);
	end

	disp(cls);
	ap = VOCap(recall, precision);
	fprintf('AP = %.4f\n', ap);

	aa = VOCap(recall, accuracy);
	fprintf('AA = %.4f\n', aa);	
	
	fprintf('MedErr = %.4f\n', median(err));
end

function az = get_azimuth(y)
[az, el, ct] = get_angles(y);
if(az<0), az = az + 360; end	% correction to bring everything to [0, 360]

function [az, el, ct] = get_angles(y)
eps = 1e-10;
proj = [0, 0, 0, 0, 0, 1, 0, -1, 0; 0, 0, -1, 0, 0, 0, 1, 0, 0; 0, 1, 0, -1, 0, 0, 0, 0, 0];
t = norm(y);
v = y/max(t, eps);
sv = reshape(v * proj, [3, 3]);
R = eye(3) + sin(t) * sv + (1-cos(t)) * sv * sv;
el = sign(-R(2, 3)) * acosd(R(3, 3));
if el ~= 0
	az = atan2d(R(3, 1) / sind(el), R(3, 2) / sind(el));
	ct = atand(-R(1, 3) / R(2, 3));
else
	az = atan2d(R(2, 1), R(1, 1));
end
if(isnan(az)), keyboard; end

function ind = find_interval(azimuth, a)
% find what bin the azimuth lies in
for i = 1:numel(a)
    if azimuth < a(i)
        break;
    end
end
ind = i - 1;
if azimuth > a(end)
    ind = 1;
end
