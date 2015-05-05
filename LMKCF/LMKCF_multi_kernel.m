function LMKCF_multi_kernel(dataset, kernel_type, nRepeat)
data_dir = fullfile(pwd, '..', 'data');
kernel_dir = fullfile(data_dir, [dataset, '_kernel']);
file_list = dir(kernel_dir);

kernel_list = {};
iKernel = 0;
for iFile = 1:length(file_list)
    sName = file_list(iFile).name;
    if (~strcmp(sName, '.') && ~strcmp(sName, '..'))
        if ~isempty(kernel_type)
            for iType = 1:length(kernel_type)
                if ~isempty(strfind(sName, kernel_type{iType}))
                    iKernel = iKernel + 1;
                    kernel_list{iKernel} = sName; %#ok<AGROW>
                end
            end
        else
            iKernel = iKernel + 1;
            kernel_list{iKernel} = sName; %#ok<AGROW>
        end
    end
end

load(fullfile(data_dir, dataset), 'y');
if ~exist('y', 'var')
    error(['y is not found in ', dataset]);
end
nClass = length(unique(y));

res_dir = fullfile(pwd, [dataset, '_res']);
if ~exist(res_dir, 'dir')
    mkdir(res_dir);
end

nKernel = length(kernel_list);
Ks = zeros(length(y), length(y), nKernel);

for iKernel = 1:length(kernel_list)
    iFile = kernel_list{iKernel};
    
    clear K;
    
    load(fullfile(kernel_dir, iFile), 'K');
    Ks(:,:,iKernel) = K;
end
clear K;

lmkcf_res = [];
obj_final = [];
kw_aio = cell(nRepeat, 1);
disp(['Total number of Kernels: ', num2str(length(kernel_list)) '!']);

disp(['LMKCF on multi kernel begin ...']);
lmkcf_res_file = fullfile(res_dir, [dataset, '_res_lmkcf.mat']);
if exist(lmkcf_res_file, 'file')
    load(lmkcf_res_file, 'res_lmkcf_aio');
else
    rng('default');
    for iRepeat = 1:nRepeat
        t_start = clock;
        disp(['LMKCF ',  num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations begin ...']);

        [~, V, w_final, nIter_final, objhistory_final] = LMKCF_Multi(Ks, nClass, struct('maxIter', [], 'nRepeat', 1), [], []);
        label_lmkcf = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 20);
        lmkcf_res = [lmkcf_res; ClusteringMeasure(y, label_lmkcf)];%#ok<AGROW>
        obj_final = [obj_final; objhistory_final(end)];%#ok<AGROW>
        kw_aio{iRepeat} = w_final;
        t_end = clock;
        disp(['LMKCF ',  num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations done.']);
        disp(['LMKCF exe time: ', num2str(etime(t_end, t_start))]);
    end
    if size(lmkcf_res, 1) > 1
        [~, minIdx] = min(obj_final);
        lmkcf_res_obj = lmkcf_res(minIdx,:);
        kw_obj = kw_aio{iRepeat};
        lmkcf_res_mean = mean(lmkcf_res);
    else
        lmkcf_res_mean = lmkcf_res;
    end
    save(lmkcf_res_file,  'lmkcf_res_mean');
    disp(['LMKCF on multi kernel done']);
    
    res_lmkcf_aio = lmkcf_res_mean;
    
    clear Ks K lmkcf_res_mean lmkcf_res_obj;
    save(fullfile(res_dir, [dataset, '_res_lmkcf_multi_kernel.mat']), 'res_lmkcf_aio', 'kernel_list');
end

end