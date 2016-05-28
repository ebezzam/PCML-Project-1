function [ L ] = computeCostRMSE( y, tX, beta )
%COMPUTECOSTMSE compute MSE cost for given input, output and model parameters
% Input:
% y - (Nx1) output vector
% tX - Nx(D+1) input vector, first column is 1 for bias term
% beta - MSE coefficients
%
% Output:
% L - cost for given input, ouput and model parameters

    N = length(y);
    e = y - tX*beta;    % compute error
    L = sqrt(e'*e/(N));     % compute MSE

end

