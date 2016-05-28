function [ beta ] = leastSquaresGD( y, tX, alpha )
%LEASTSQUARESGD Least squares using gradient descent
% Input:
% y - (Nx1) output vector
% tX - Nx(D+1) input vector, first column is 1 for bias term
% alpha - step size paramter
%
% Output:
% beta - optimal coefficients for least square, computed using gradient
% descent

    % parameters
    D = size(tX,2); % number of dimensions
    maxIters = 2000;

    % initialize beta
    beta = zeros(D,1);

    % iterate
    for k = 1:maxIters
        g = computeGradientMSE(y, tX, beta);
        beta = beta - alpha * g;
        if g'*g < 1e-5; break; end;
    end
end

function [ g ] = computeGradientMSE( y, tX, beta )
%COMPUTEGRADIENTMSE compute gradient for MSE
    N = length(y);
    e = y - tX*beta;    % compute error
    g = (-1)*tX'*e/N;   % compute gradient
end

