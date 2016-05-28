function [ g1, g2, g3 ] = assignGroups( mu, x1, x2 )
%ASSIGNGROUPS assign groups according to mean

    % group vectors
    N = length(x2);
    K = size(mu,1);
    g1 = [];
    g2 = [];
    g3 = [];
    dist = zeros(size(mu,1),1);
    for n = 1:N
        xn = [x1(n) x2(n)];
        for k = 1:K
           dist(k) = norm(xn-mu(k,:));
        end
        [~, group] = min(dist);
        if(group==1)
            g1 = [g1 n];
        end
        if(group==2)
            g2 = [g2 n];
        end
        if(group==3)
            g3 = [g3 n];
        end

    end

%     g1 = find(x2<0.5);
%     x1(g1) = [];
%     g2 = find(x1<1);
%     g3 = find(x1>=1);
    

end

