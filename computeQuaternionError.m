function theta = computeQuaternionError(ytest, yhat)
% function to compute viewpoint angle between the gt and predicted pose
theta = 2*acosd(abs(sum(ytest.*yhat, 2)));
