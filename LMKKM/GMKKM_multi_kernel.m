function GMKKM_multi_kernel(dataset, kernel_type, nRepeat)
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

gmkkm_res = [];
obj_final = [];
kw_aio = cell(nRepeat, 1);
disp(['Total number of Kernels: ', num2str(length(kernel_list)) '!']);

disp(['GMKKM on multi kernel begin ...']);
gmkkm_res_file = fullfile(res_dir, [dataset, '_res_gmkkm.mat']);
if exist(gmkkm_res_file, 'file')
    load(gmkkm_res_file, 'res_gmkkm_aio');
else
    rng('default');
    for iRepeat = 1:nRepeat
        t_start = clock;
        disp(['GMKKM ',  num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations begin ...']);

        parameters = struct();
        parameters.cluster_count = nClass;
        parameters.iteration_count = 10;

        state = mkkmeans_train(Ks, parameters);
        gmkkm_res = [gmkkm_res; ClusteringMeasure(y, state.clustering)];%#ok<AGROW>
        obj_final = [obj_final; state.objective(end)];%#ok<AGROW>
        kw_aio{iRepeat} = state.theta;
        t_end = clock;
        disp(['GMKKM ',  num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations done.']);
        disp(['GMKKM exe time: ', num2str(etime(t_end, t_start))]);
    end
    if size(gmkkm_res, 1) > 1
        [~, minIdx] = min(obj_final);
        gmkkm_res_obj = gmkkm_res(minIdx,:);
        kw_obj = kw_aio{iRepeat};
        gmkkm_res_mean = mean(gmkkm_res);
    else
        gmkkm_res_mean = gmkkm_res;
    end
    save(gmkkm_res_file,  'gmkkm_res_mean');
    disp(['GMKKM on multi kernel done']);
    
    res_gmkkm_aio = gmkkm_res_mean;
    
    clear Ks K gmkkm_res_mean gmkkm_res_obj;
    save(fullfile(res_dir, [dataset, '_res_gmkkm_multi_kernel.mat']), 'res_gmkkm_aio', 'kernel_list');
end

end