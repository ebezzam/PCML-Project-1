function [ degreeStar, lambdaStar, errTrain, errTest ] = optimizeClassDegreeLambda( y, X, alpha, degreeVals, regVals, nCont, K )
%OPTIMIZEDEGREELAMBDA use k-fold cross validation to find the optimal value for
%lambda and multinomial degree in ridge regression

    errTr = ones(length(degreeVals), length(regVals));
    errTe = ones(length(degreeVals), length(regVals));
    
    % split data in K fold (we will only create indices)
    setSeed(1);
    N = size(y,1);
    idx = randperm(N);
    Nk = floor(N/K);
    for k = 1:K
        idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
    end
    
    for i = 1:length(degreeVals)
        degree = degreeVals(i);
        
        for j = 1:length(regVals)
            lambda = regVals(j);
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
                % normalize test data according to stats of training data
                N_te = size(XTe, 1);
                XTe_n = (XTe(:,1:nCont) - repmat(uX,N_te,1))./repmat(stdX,N_te,1);
                % create polynomial input vector
                XTr_poly = [myPoly(XTr_n,degree), XTr(:,(nCont+1):end)];
                XTe_poly = [myPoly(XTe_n,degree), XTe(:,(nCont+1):end)];
                % add bias term
                tXTr = [ones(N_tr,1) XTr_poly];
                tXTe = [ones(N_te,1) XTe_poly];
                % compute model using training data
                [ beta ] = penLogisticRegression( yTr, tXTr, alpha, lambda );
                % calculate error
                errTrSub(k) = computeCostZeroOne( yTr, tXTr, beta );
                errTeSub(k) = computeCostZeroOne( yTe, tXTe, beta );
            end
            errTr(i,j) = mean(errTrSub);
            errTe(i,j) = mean(errTeSub);
        end
    end
    
    % find degree and lambda that yields minimum degree and lambda    
    vals1 = min(errTe, [], 2);           % search for minimum error for each degree
    [~, degreeStarIdx] = min(vals1);        % search for degree that gives lowest possible error
    [~, lambdaStarIdx] = min(errTe(degreeStarIdx,:));
    lambdaStar = regVals(lambdaStarIdx);
    degreeStar = degreeVals(degreeStarIdx);
    
    % return errors corresponding to lowest degree
%     errTrain = errTr(degreeStar,:);
%     errTest = errTe(degreeStar,:);
    
    % return errors corresponding to lowest lambda
%     errTrain = errTr(:,lambdaStarIdx);
%     errTest = errTe(:,lambdaStarIdx);
    
    % return lowest errors for each degree
    errTrain = min(errTr, [], 2); 
    errTest = min(errTe, [], 2); 

end

