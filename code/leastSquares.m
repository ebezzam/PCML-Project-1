function [ beta ] = leastSquares( y, tX )
%LEASTSQUARES Least squares using normal equations
% Input:
% y - (Nx1) output vector
% tX - Nx(D+1) input vector, first column is 1 for bias term
%
% Output:
% beta - optimal coefficients for least square, computed using normal
% equations

    beta = ((tX'*tX)\tX') * y;

end

