function computeARP(filename, dets_path)
% function to compute the detection metrics

% relevant paths
pascal3d_path = 'data/pascal3d/';
anno_path = fullfile(pascal3d_path, 'Annotations');

classes = {'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', ...
	'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor'};
num_classes = length(classes);

% load list of all images
tmp = load(fullfile(dets_path, 'dbinfo'));
image_names = tmp.image_names;
N = length(image_names);

% load predictions
tmp = load(fullfile('results', filename));
boxes_all = tmp.bbox;
ypred_all = tmp.ypred;
labels_all = tmp.labels;

total_all = zeros(1, num_classes);
correct_all = zeros(1, num_classes);
correct_view_all = zeros(1, num_classes);
medErr_all = zeros(1, num_classes);

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
		anno_file = fullfile(anno_path, sprintf('%s_pascal', cls), [image_name, '.mat']);
		if ~exist(anno_file, 'file'), continue; end
		% load annotation
		tmp = load(anno_file);
		record = tmp.record;
		objects = record.objects;
		% check cls
		clsinds = strcmp(cls, {objects(:).class});
		diff = [objects(:).difficult];
		clsinds = find(clsinds & ~diff);
		% get gt-bbox and gt-pose
		n = numel(clsinds);
		bbox = zeros(n, 4); view_gt = zeros(n, 3); view_az = zeros(n, 1);
		for j = 1:n
			bbox(j, :) = objects(clsinds(j)).bbox;
			viewpoint = objects(clsinds(j)).viewpoint;
			if viewpoint.distance == 0
				az = viewpoint.azimuth_coarse;
				el = viewpoint.elevation_coarse;
				ct = viewpoint.theta;
			else
				az = viewpoint.azimuth;
				el = viewpoint.elevation;
				ct = viewpoint.theta;
			end
			view_gt(j, :) = get_v(get_R(az, el, ct));
			view_az(j) = az;
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
			view_pr = ypred(j, :);
			
			if ~isempty(bbox)
				o = box_overlap(bbox, bbox_pr);
				[maxo, index] = max(o);
				if maxo >= 0.5 && det(index)==0
					overlap{num_pr} = index;
					correct(num_pr) = 1;
					det(index) = 1;
					theta = computeGeodesicError(view_gt(index, :), view_pr);
					err = [err, theta];
					if theta < 30
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

	fprintf('Stats: \t num_total=%d \t percent_correct=%0.2f \t percent_correct_view:%0.2f \t MedErr = %2.1f \n', sum(count), num_correct/sum(count), num_correct_view/sum(count), median(err));
	
	total_all(cls_id) = sum(count);
	correct_all(cls_id) = num_correct;
	correct_view_all(cls_id) = num_correct_view;
	medErr_all(cls_id) = median(err);
end
