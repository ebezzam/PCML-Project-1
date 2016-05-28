function [ lambdaStar, errTr, errTe ] = optimizeLambda( y, X, vals, nCont, K )
%OPTIMIZELAMBDA use k-fold cross validation to find the optimal value for
%lambda in ridge regression

    errTr = ones(length(vals),1);
    errTe = ones(length(vals),1);
    
    % split data in K fold (we will only create indices)
    setSeed(1)
    N = size(y,1);
    idx = randperm(N);
    Nk = floor(N/K);
    for k = 1:K
        idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
    end
    
    for i = 1:length(vals)
        lambda = vals(i);
        
        % implement k-folds
        errTrSub = ones(K,1);
        errTeSub = ones(K,1);
        for k = 1:K
            
            % get k'th subgroup in test, others in train
            idxTe = idxCV(k,:);
            idxTr = idxCV([1:k-1 k+1:end],:);
            idxTr = idxTr(:);
            yTe = y(idxTe);
            XTe = X(idxTe,:);
            yTr = y(idxTr);
            XTr = X(idxTr,:);
            
            % normalize continuous features training
            N_tr = size(XTr, 1);
            [ XTr_n, uX, stdX ] = normalizeFeatures( XTr(:,1:nCont) );
            XTr_n = [XTr_n XTr(:,(nCont+1):end)];

            % normalize test data according to stats of training data
            N_te = size(XTe, 1);
            XTe_n = (XTe(:,1:nCont) - repmat(uX,N_te,1))./repmat(stdX,N_te,1);
            XTe_n = [XTe_n XTe(:,(nCont+1):end)];
            
            % add bias term
            tXTr = [ones(N_tr,1) XTr_n];
            tXTe = [ones(N_te,1) XTe_n];
            
            % compute model using training data
            [ beta ] = ridgeRegression( yTr, tXTr, lambda );
            
            % calculate error
            errTrSub(k) = computeCostRMSE( yTr, tXTr, beta );
            errTeSub(k) = computeCostRMSE( yTe, tXTe, beta );
        end
        errTr(i) = mean(errTrSub);
        errTe(i) = mean(errTeSub);
    end
    [~, lambdaStarIdx] = min(errTe);
    
    lambdaStar = vals(lambdaStarIdx);

end

