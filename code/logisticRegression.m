function [ beta ] = logisticRegression( y, tX, alpha )
%LOGISTICREGRESSION Logistic regression using gradient descent or Newton's
%method
% Input:
% y - (Nx1) output vector
% tX - Nx(D+1) input vector, first column is 1 for bias term
% alpha - step size (in case of gradient descent)
%
% Output:
% beta - optimal coefficients for logistic regression, computed using
% gradient descent of Newton's method

    % parameters
    D = size(tX,2); % number of dimensions
    maxIters = 10000;

    % initialize beta
    beta = zeros(D,1);

    % iterate
    for k = 1:maxIters
        g = computeGradientLR(y, tX, beta);

%         % gradient descent
%         beta = beta + alpha * g;    % apply gradient descent

        % newton's method
        H = computeHessian(tX, beta);
        beta = beta + alpha * (H\g);

        % break if minimum reached
        if g'*g < 1e-5 
            break;
        end;
    end
end

function [ g ] = sigmoid( x )
    g = zeros(length(x),1);
    g(x > 0) = 1 ./(1 + exp(-1 .* x (x > 0))) ;
    g(x <= 0) = exp(x(x <= 0 )) ./ (1 + exp( x(x <= 0)));
end

function [ g ] = computeGradientLR( y, tX, beta )
%COMPUTEGRADIENTLR compute gradient for logistic regression
    h = sigmoid(tX*beta);
    g = -tX'*(h-y);
end

function [ H ] = computeHessian( tX, beta )
%COMPUTEHESSIAN compute derivative of the gradient in order to obtain
%Hessian
    N = size(tX,1);
    S = zeros(N,N);
    for i = 1:N
       h = sigmoid(tX(i,:)*beta);
       S(i,i) = h*(1-h);
    end
    H = tX'*S*tX;
end

