clc;
clear all;
m_degree=1.08;
threshold=1e-5;
repeat_num=20;

%read data
data=load('dataset/data.txt');
ground_truth=load('dataset/label.txt');
ground_truth=ground_truth+1;
n=hist(ground_truth,min(ground_truth):1:max(ground_truth));
cluster=size(find(n),2);
%normalize data
data=data_normalize(data,'var');
data=normr(data);
%compute kvalue
kvalue=compute_kvalue(data);
number=size(kvalue,1);
sizeK=size(kvalue, 3);

Ks = kvalue;
gnd = ground_truth;
m_degree_candi = [1.02:0.02:1.1, 1.15:0.05:1.2, 1.3:0.1:1.5];
nRepeat = 10;

[res_gmkfcm, res_gmkfcm_detail] = GMKFCM_multi_kernel(Ks, cluster, gnd, m_degree_candi, nRepeat);
[res_lmkfcm, res_lmkfcm_detail] = LMKFCM_multi_kernel(Ks, cluster, gnd, m_degree_candi, nRepeat);
%%
result_Kmean=zeros(1,repeat_num);
result_FCM=zeros(1,repeat_num);
result_KFC=zeros(sizeK,repeat_num);
result_KFC2=zeros(sizeK,repeat_num);
result_MKFC=zeros(1,repeat_num);
result_GMKFCM=zeros(1,repeat_num);
result_LMKFCM=zeros(1,repeat_num);

% t1 = tic;
% for iter=1:repeat_num
%     U=initfcm(cluster, number);
%     U_=U';
%     for i=1:size(kvalue,3)
%         U_tmp = KFC(U_,cluster,m_degree,threshold,kvalue(:,:,i));
%         [t,index]=sort(U_tmp,2);
%         dx = index(:,end);
%         result_KFC(i,iter)=nmi(dx,ground_truth);
%     end
% end
% t2 = toc(t1);
% t11 = tic;
% for iter=1:repeat_num
%     U=initfcm(cluster, number);
%     U_=U';
%     for i=1:size(kvalue,3)
%         U_tmp = KFCM(kvalue(:,:,i), cluster, m_degree, struct('maxIter', 100), U_);
%         [t,index]=sort(U_tmp,2);
%         dx = index(:,end);
%         result_KFC2(i,iter)=nmi(dx,ground_truth);
%     end
% end
% t22 = toc(t11);
% t  = [mean(result_KFC,2), mean(result_KFC2,2)];
% error = sum(abs(t(:,1) - t(:,2))) / size(t,1);

t1 = tic;
for iter=1:repeat_num
    U=initfcm(cluster, number);
    U_=U';

    [U_tmp weight]= GMKFCM(kvalue, cluster, m_degree, struct('maxIter', [], 'nRepeat', 10),U_);
    [t,index]=sort(U_tmp,2);
    dx = index(:,end);
    result_GMKFCM(1,iter)=nmi(dx,ground_truth);
    
    [U_tmp weight]= LMKFCM(kvalue, cluster, m_degree, struct('maxIter', [], 'nRepeat', 10),U_);
    [t,index]=sort(U_tmp,2);
    dx = index(:,end);
    result_LMKFCM(1,iter)=nmi(dx,ground_truth);    
end
tg = toc(t1);

% t1 = tic;
% for iter=1:repeat_num
%     U=initfcm(cluster, number);
%     U_=U';
% 
%     [U_tmp weight]= LMKFCM(kvalue, cluster, m_degree, struct('maxIter', []),U_);
%     [t,index]=sort(U_tmp,2);
%     dx = index(:,end);
%     result_LMKFCM(1,iter)=nmi(dx,ground_truth);
% end
% tl = toc(t1);
%%
for iter=1:repeat_num
    fprintf('iter %d\n',iter);    
    %kmeans
    dx = kmeans(data,cluster,'EmptyAction','drop','Replicates',50);
    result_Kmean(1,iter)=nmi(dx,ground_truth);
    %initialize U
    U=initfcm(cluster, number);
    %fcm
    U_tmp=FCM(U,data,cluster,m_degree,threshold);
    [t,index]=sort(U_tmp,2);
    dx = index(:,end);
    result_FCM(1,iter)=nmi(dx,ground_truth);

    U_=U';
    %kfc
    for i=1:size(kvalue,3)
        U_tmp = KFC(U_,cluster,m_degree,threshold,kvalue(:,:,i));
        [t,index]=sort(U_tmp,2);
        dx = index(:,end);
        result_KFC(i,iter)=nmi(dx,ground_truth);
    end
    
    for i=1:size(kvalue,3)
        U_tmp = KFCM(kvalue(:,:,i), cluster, m_degree, struct('error', threshold), U_);
        [t,index]=sort(U_tmp,2);
        dx = index(:,end);
        result_KFC2(i,iter)=nmi(dx,ground_truth);
    end
    %mkfc
%     [U_tmp weight]= MKFC(U_,cluster,m_degree,threshold,kvalue);
%     [t,index]=sort(U_tmp,2);
%     dx = index(:,end);
%     result_MKFC(1,iter)=nmi(dx,ground_truth);
end
fprintf('Kmean : %4f\n',mean(result_Kmean));
fprintf('FCM : %4f\n',mean(result_FCM));
% for i=1:size(kvalue,3)
%     fprintf('KFC%d : %4f\n',i,mean(result_KFC(i,:)));
% end
% fprintf('MKFC : %4f\n', mean(result_MKFC));
% fprintf('weight : %4f\n',weight);
% clear all;
% clc;
