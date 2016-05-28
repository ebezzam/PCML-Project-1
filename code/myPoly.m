function Xpoly = myPoly(X,degree)
% build matrix Phi for polynomial regression of a given degree
    Xpoly = X;
    for k = 2:degree
        Xpoly = [Xpoly X.^k];
    end
end

