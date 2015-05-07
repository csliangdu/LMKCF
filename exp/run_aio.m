clear;clc;

addpath(fullfile(pwd, '..', 'lib'));

ds = {'ORL_400n_1024d_40c_zscore_uni', 'AR_840n_768d_120c_uni',...
    'COIL20_1440n_1024d_20c', 'binaryalphadigs_1404n_320d_36c',...
    'medical_706n_1449d_17c', 'bbcnewssport_737n_1000d_5c_tfidf_uni',...
    'hitech_2301n_22498d_6c_tfidf_uni', 'k1b_2340n_21839d_6c_tfidf_uni',...
    'news4a_3840n_4989d_4c_tfidf_uni', 'webkb_4199n_1000d_4c_tfidf_uni',...
    };

for i =[11]%1:length(ds)
    run_single_dataset(ds{i});
end

res_acc_aio = [];
res_nmi_aio = [];
res_purity_aio = [];
for iData = 1:length(ds)
    clear table_acc table_nmi table_purity
    load(fullfile([ds{iData} '_res'], [ds{iData} '_res_table.mat']));
    res_acc_aio = [res_acc_aio; table_acc];
    res_nmi_aio = [res_nmi_aio; table_nmi];
    res_purity_aio = [res_purity_aio; table_purity];
end

idx = [1:size(res_acc_aio,2)];
res_acc_aio = res_acc_aio(:, idx);
res_nmi_aio = res_nmi_aio(:, idx);
res_purity_aio = res_purity_aio(:, idx);

res_lmkcf_aio = [];
for i1 = 1:size(res_purity_aio,1)
    res_lmkcf_aio = [res_lmkcf_aio; res_acc_aio(i1,:)];
    res_lmkcf_aio = [res_lmkcf_aio; res_nmi_aio(i1,:)];
    res_lmkcf_aio = [res_lmkcf_aio; res_purity_aio(i1,:)];
end
res_sort_val = sum(res_lmkcf_aio);
res_lmkcf_aio = [res_lmkcf_aio; mean(res_acc_aio)];
res_lmkcf_aio = [res_lmkcf_aio; mean(res_nmi_aio)];
res_lmkcf_aio = [res_lmkcf_aio; mean(res_purity_aio)];

z = zeros(1, size(res_lmkcf_aio,2));
[~, idx] = sort(res_sort_val, 'descend');
z(idx) = [1:length(idx)];
res_lmkcf_aio = [res_lmkcf_aio; z];

save('lmkcf_res_aio.mat', 'res_acc_aio', 'res_nmi_aio', 'res_purity_aio', 'res_lmkcf_aio',  'ds');

rowLabels = {'ORL', 'ORL','ORL', 'AR', 'AR', 'AR', ...
'COIL20', 'COIL20', 'COIL20', 'BA', 'BA', 'BA', ...
    'medical', 'medical', 'medical', 'bbcnews', 'bbcnews', 'bbcnews', ...
    'hitech', 'hitech', 'hitech', 'k1b', 'k1b', 'k1b', ...
    'news4a', 'news4a', 'news4a', 'webkb', 'webkb', 'webkb', ...
    'Average', 'Average', 'Average', 'Rank'};
columnLabels = {'KKMb', 'KKMa', 'SCb', 'SCa', 'KCFb', 'KCFa', 'LCCFb', 'LCCFa', 'MACFb', 'MACFa', 'Coreg', 'LMKKM', 'RMKKM', 'GMKCF','LMKCF'};
matrix2latex(res_lmkcf_aio, 'lmkcf_res_aio.tex', 'rowLabels', rowLabels, 'columnLabels', columnLabels, 'alignment', 'c', 'format', '%4.4f','size', 'tiny');
rmpath(fullfile(pwd, '..', 'lib'));