addpath('C:\Program Files\Mosek\7\toolbox\r2013a')
p=10;
n=4000;
rand('state',10);
for i=1:p
    A=rand(n,n);
    Ks{i}=A*A';
end

theta=rand(n,p);
tic
[Theta,objHis]=APG(Ks,theta);
toc
objHis(end)

%  Q = sparse(zeros(n * p, n * p));
%  for m = 1:p
%      start_index = (m - 1) * n + 1;
%      end_index = m * n;
%      Q(start_index:end_index, start_index:end_index) = Ks{m};
%  end
%  tic
%  res = mskqpopt(Q, zeros(n * p, 1), repmat(eye(n, n), 1, p), ones(n, 1), ones(n, 1), zeros(n * p, 1), ones(n *p, 1), [], 'minimize echo(0)');
%  toc
%  newTheta = reshape(res.sol.itr.xx, n, p);
%  obj=0;
%  for i=1:p
%      obj=obj+newTheta(:,i)'*Ks{i}*newTheta(:,i);
%  end
%  obj

