function [ X_n, uX, stdX ] = normalizeFeatures( X )
%NORMALIZEFEATURES normalize feature values
% Input:
% X - (NxD)input values (can be matrix or vector)
%
% Outputs:
% X_n - normalized input values
% uX - mean of each dimension
% stdX - standard deviation of each dimension

    N = size(X,1);
    uX = mean(X);
    stdX = std(X);
    X_n = (X - repmat(uX,N,1))./repmat(stdX,N,1);

end

