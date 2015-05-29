function [idx, weight, obj_final]=MKFC(U,cluster,degree,error,kvalue)
% Input
%       U : initial membership matrix
% cluster : desired number of clusters
%  degree : fuzzification degree
%   error : stop threshold
%  kvalue : N x N x k affinity matrices
% Output
% U_final : membership matrix
%  weight : weight assignment to affinity matrices
data_n = size(kvalue, 1);
dimension=size(kvalue,3);
% Main loop
U_final=U;
for iter=1:100
    bk=zeros(1,dimension);
    alpha=zeros(data_n,cluster,dimension);
    %calculate normalized memberships
    mf = U_final.^degree;
    mf_tmp = mf*diag(1./sum(mf));
    for k=1:dimension
        alpha(:,:,k)=ones(data_n,1)*diag(mf_tmp'*kvalue(:,:,k)*mf_tmp)'-2*kvalue(:,:,k)*mf_tmp+1;
    end
    %calculate coefficients bk
    for k=1:dimension
        bk(1,k)=sum(sum(mf.*alpha(:,:,k)));
    end
    %update weights w
    w=ones(1,dimension)./bk;
    w=w/sum(w);
    %calculate distances D
    dist=zeros(data_n,cluster);
    wtmp=w.^2;
    for k=1:dimension
        dist=dist+alpha(:,:,k)*wtmp(1,k);
    end
    tmp = dist.^(-1/(degree-1));
    %update memberships U
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
obj_final = obj_fcn(end);
[~, index]=sort(U_final, 2);
idx = index(:,end);
weight=w;
clear w;
clear mf;
clear mf_tmp;
clear tmp;
clear obj_fcn;
clear dist;
clear bk;
