function [ L ] = computeCostLogLoss( y, tX, beta )
%COMPUTELOGLOSS compute negative of log likelihood

    N = length(y);
    pHatN = sigmoid(tX*beta); 
    sum = 0;
    for i=1:N
        sum = sum + y(i)*log(pHatN(i)) + (1-y(i))*log(1-pHatN(i));
    end
    L = (-1/N) * sum;
    
end

