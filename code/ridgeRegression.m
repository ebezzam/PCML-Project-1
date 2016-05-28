function [ beta ] = ridgeRegression( y, tX, lambda )
%RIDGEREGRESSION compute optimal model parameters for ridge regression
%using normal equation
% Input:
% y - (Nx1) output vector
% tX - Nx(D+1) input vector, first column is 1 for bias term
% lambda - the regularization coefficient
%
% Output:
% beta - optimal coefficients for regularized linear regression, computed using normal
% equations
    
    Im = eye(size(tX,2));
    Im(1,1) = 0;    % beta_0 (bias term coefficient) should not be penalized
    beta = ((tX'*tX + lambda*Im)\tX') * y;

end

