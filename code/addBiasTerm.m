function [ tX ] = addBiasTerm( X )
%ADDBIASTERM add columns of ones

    N = size(X,1);
    tX = [ones(N,1) X];

end

