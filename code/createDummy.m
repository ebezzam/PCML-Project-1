function [ X_d ] = createDummy( X )
%CREATEDUMMY create dummy variables for each column consisting of a
%categorical variable

X_d = [];
D = size(X,2);
N = size(X,1);

for d = 1:D
    s = X(:,d);
    nCategories = max(s);
    X_temp = zeros(N,nCategories);
    for val = 1:nCategories
        X_temp(find(s==val),val) = 1;
    end
    X_d = [X_d X_temp];
end
% 
% for d = 1:D
%     s = X(:,d);
%     nCategories = max(s);
%     X_temp = zeros(N,nCategories+1);
%     for val = 0:nCategories
%         X_temp(find(s==val),val+1) = 1;
%     end
%     X_d = [X_d X_temp];
% end

end

