function U_final=FCM(U,data,cluster,degree,error)
% Input 
%       U : initial membership matrix
%    data : a set of data points
% cluster : desired number of clusters
%  degree : fuzzification degree 
%   error : stop threshold  
% Output
% U_final : membership matrix 
    % Main loop
    U_final=U;
    for iter=1:100
        mf = U_final.^degree; % mf = U.^expo; MF matrix after exponential modification
        center = mf*data./((ones(size(data, 2), 1)*sum(mf'))'); % center = mf*data./((ones(size(data, 2), 1)*sum(mf'))'); % new center
        % fill the distance matrix
        dist = zeros(size(center, 1), size(data, 1));
        if size(center, 2) > 1,
            for k = 1:size(center, 1),
                dist(k, :)= sqrt(sum(((data-ones(size(data, 1), 1)*center(k, :)).^2)'));
            end
        else	% 1-D data
            for k = 1:size(center, 1),
                dist(k, :)= abs(center(k)-data)';
            end
        end
        tmp = dist.^(-2/(degree-1));    % tmp = dist.^(-2/(expo-1)); calculate new U, suppose degree != 1
        U_final = tmp./(ones(cluster, 1)*sum(tmp));%U_new = tmp./(ones(cluster_n, 1)*sum(tmp));
        obj_fcn(iter) = sum(sum((dist.^2).*mf));% obj_fcn = sum(sum((dist.^2).*mf));
        % check termination condition
        if iter > 1
            if abs(obj_fcn(iter) - obj_fcn(iter-1))< error
                break; 
            end
        end
    end
    U_final=U_final';
    clear mf;
    clear center;
    clear tmp;
    clear obj_fcn;
    clear dist;
    