function LMKKM_multi_kernel(dataset, kernel_type, nRepeat)
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

lmkkm_res = [];
obj_final = [];
kw_aio = cell(nRepeat, 1);
disp(['Total number of Kernels: ', num2str(length(kernel_list)) '!']);

disp(['LMKKM on multi kernel begin ...']);
lmkkm_res_file = fullfile(res_dir, [dataset, '_res_lmkkm.mat']);
if exist(lmkkm_res_file, 'file')
    load(lmkkm_res_file, 'res_lmkkm_aio');
else
    rng('default');
    for iRepeat = 1:nRepeat
        t_start = clock;
        disp(['LMKKM ',  num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations begin ...']);

        parameters = struct();
        parameters.cluster_count = nClass;
        parameters.iteration_count = 10;

        state = lmkkmeans_train(Ks, parameters);
        lmkkm_res = [lmkkm_res; ClusteringMeasure(y, state.clustering)];%#ok<AGROW>
        obj_final = [obj_final; state.objective(end)];%#ok<AGROW>
        kw_aio{iRepeat} = state.Theta;
        t_end = clock;
        disp(['LMKKM ',  num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations done.']);
        disp(['LMKKM exe time: ', num2str(etime(t_end, t_start))]);
    end
    if size(lmkkm_res, 1) > 1
        [~, minIdx] = min(obj_final);
        lmkkm_res_obj = lmkkm_res(minIdx,:);
        kw_obj = kw_aio{iRepeat};
        lmkkm_res_mean = mean(lmkkm_res);
    else
        lmkkm_res_mean = lmkkm_res;
    end
    save(lmkkm_res_file,  'lmkkm_res_mean');
    disp(['LMKKM on multi kernel done']);
    
    res_lmkkm_aio = lmkkm_res_mean;
    
    clear Ks K lmkkm_res_mean lmkkm_res_obj;
    save(fullfile(res_dir, [dataset, '_res_lmkkm_multi_kernel.mat']), 'res_lmkkm_aio', 'kernel_list');
end

end