function U_final =KFC(U,cluster,degree,error,kvalue)
% Input 
%       U : initial membership matrix
% cluster : desired number of clusters
%  degree : fuzzification degree 
%   error : stop threshold  
%  kvalue : N x N affinity matrix
% Output
% U_final : membership matrix
    data_n = size(kvalue, 1);
    % Main loop
    U_final=U;
    for iter=1:100
        %calculate normalized memberships
        mf = U_final.^degree;
        mf_tmp = mf*diag(1./sum(mf));
        %calculate distances D
        dist=ones(data_n,1)*diag(mf_tmp'*kvalue*mf_tmp)'-2*kvalue*mf_tmp+1;
        %update memberships U
        tmp = dist.^(-1/(degree-1));    
        U_final = tmp./(sum(tmp,2)*ones(1,cluster)); 

        % objective function
        obj_fcn(iter) = sum(sum(dist.*mf));
        U_old=U_final;
        % check termination condition
        if iter > 1
            if abs(obj_fcn(iter) - obj_fcn(iter-1))< error
                break; 
            end
        end
    end
    clear mf;
    clear mf_tmp;
    clear tmp;
    clear obj_fcn;
    clear dist;
    