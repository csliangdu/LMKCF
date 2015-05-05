function [ Theta,objHistory ] = QP_APG( Ks,X )
%APG 此处显示有关此函数的摘要
%   此处显示详细说明

% p=length(Ks);
% n=size(Ks{1},1);
[n, ~, p] = size(Ks);

mu = 1.05;
Lf = 10000; % intial value of Lipschitz constant
maxiter_pg = 100;


objHistory = [];

iter = 0;

delta=[];

while  iter < maxiter_pg
    %%----------------------------------------------------------
    % search the valid Lipschitz constant, which satisfies
    %   Obj( proj(X_old, Lf)) < Obj_pg( proj(X_old, Lf), X_old)
    %%----------------------------------------------------------
    Lf_candi = Lf;
    is_valid_Lf = true;
    % extract non-loop parts
    obj_pg_1=0;
    grad_X=zeros(n,p);
    for i=1:p
        grad_X(:,i)=Ks(:,:,i)*X(:,i);
        obj_pg_1=obj_pg_1+X(:,i)'*grad_X(:,i);
    end
    
    while is_valid_Lf

        Y = X - 1 ./ Lf_candi * grad_X;
        X_candi=SimplexProj(Y); 
        obj_candi=0;
        for i=1:p
            obj_candi=obj_candi+X_candi(:,i)'*Ks(:,:,i)*X_candi(:,i);
        end

        X_diff = (X_candi - X);
        obj_pg_2 = sum(sum(X_diff .* grad_X));
        obj_pg_3 = sum(sum(X_diff .* X_diff));
        obj_pg = obj_pg_1 + obj_pg_2 + .5 * Lf_candi * obj_pg_3;
        if obj_candi > obj_pg
            Lf_candi = Lf_candi * mu;
        else
            is_valid_Lf = false;
        end
    end
    Lf = Lf_candi; % get the valid Lipschitz constant
    X = X_candi; % get the result from proximal operator with valid Lipschitz constant
    obj=0;
    for i=1:p
        obj=obj+X(:,i)'*Ks(:,:,i)*X(:,i);
    end

    objHistory = [objHistory; obj]; %#ok
     if length(objHistory)>2
         if (objHistory(end-1)-objHistory(end))/objHistory(end-1)<1e-4
             break;
         end
     end

%    epsilon = obj_pg_3; %
    iter = iter + 1;
end
%Lf
Theta=X;
end


