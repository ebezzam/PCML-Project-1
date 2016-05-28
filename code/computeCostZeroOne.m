function [ L ] = computeCostZeroOne( y, tX, beta )
%computeCostZeroOne compute error in terms of percentage
    
    error = 0;
    yHatN = sigmoid(tX*beta);
    N = length(y);
    for i = 1:N
        if(yHatN(i) <= 0.5) 
            yHatN(i) = 0; 
        else yHatN(i) = 1;
        end
        if(y(i) ~= yHatN(i)) 
            error = error + 1;
        end
    end
    L = error / N;

end

