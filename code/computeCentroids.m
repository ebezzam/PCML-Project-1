function [ mu_0, g1, g2, g3 ] = computeCentroids( x1, x2, K, mu_0 )
%COMPUTECENTROIDS Use k-means to compute centers of clusters

    maxIter = 100;

    % iterate
    nFeat = 2;
    N = length(x2);
    for i = 1:maxIter
        % assign points to clusters
        mu = zeros(K,nFeat);
        nVec = zeros(K,1);
        dist = zeros(K,1);
        for n = 1:N
            xn = [x1(n) x2(n)];
            for k = 1:K
                dist(k) = norm(xn-mu_0(k,:));
            end
            [~, group] = min(dist);
            mu(group,:) = mu(group,:) + xn;
            nVec(group) = nVec(group) + 1;
        end
        % calculate new mean for each cluster
        mu = mu./repmat(nVec,1,nFeat);

        if( abs(mu-mu_0) < 1e-05)
            break;
        end
        mu_0 = mu;
    end
    
    % group vectors
    [ g1, g2, g3 ] = assignGroups( mu_0, x1, x2 );
%     g1 = [];
%     g2 = [];
%     g3 = [];
%     for n = 1:N
%         xn = [x1(n) x2(n)];
%         for k = 1:K
%            dist(k) = norm(xn-mu_0(k,:));
%         end
%         [~, group] = min(dist);
%         if(group==1)
%             g1 = [g1 n];
%         end
%         if(group==2)
%             g2 = [g2 n];
%         end
%         if(group==3)
%             g3 = [g3 n];
%         end
%     end


end

