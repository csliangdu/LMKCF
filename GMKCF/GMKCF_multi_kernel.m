function GMKCF_multi_kernel(dataset, kernel_type, nRepeat)
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

gmkcf_res = [];
obj_final = [];
kw_aio = cell(nRepeat, 1);
disp(['Total number of Kernels: ', num2str(length(kernel_list)) '!']);

disp(['GMKCF on multi kernel begin ...']);
gmkcf_res_file = fullfile(res_dir, [dataset, '_res_gmkcf.mat']);
if exist(gmkcf_res_file, 'file')
    load(gmkcf_res_file, 'res_gmkcf_aio');
else
    rng('default');
    for iRepeat = 1:nRepeat
        t_start = clock;
        disp(['GMKCF ',  num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations begin ...']);

        [~, V, w_final, nIter_final, objhistory_final] = GMKCF_Multi(Ks, nClass, struct('maxIter', [], 'nRepeat', 1), [], []);
        label_gmkcf = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 20);
        gmkcf_res = [gmkcf_res; ClusteringMeasure(y, label_gmkcf)];%#ok<AGROW>
        obj_final = [obj_final; objhistory_final(end)];%#ok<AGROW>
        kw_aio{iRepeat} = w_final;
        t_end = clock;
        disp(['GMKCF ',  num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations done.']);
        disp(['GMKCF exe time: ', num2str(etime(t_end, t_start))]);
    end
    if size(gmkcf_res, 1) > 1
        [~, minIdx] = min(obj_final);
        gmkcf_res_obj = gmkcf_res(minIdx,:);
        kw_obj = kw_aio{iRepeat};
        gmkcf_res_mean = mean(gmkcf_res);
    else
        gmkcf_res_mean = gmkcf_res;
    end
    save(gmkcf_res_file,  'gmkcf_res_mean');
    disp(['GMKCF on multi kernel done']);
    
    res_gmkcf_aio = gmkcf_res_mean;
    
    clear Ks K gmkcf_res_mean gmkcf_res_obj;
    save(fullfile(res_dir, [dataset, '_res_gmkcf_multi_kernel.mat']), 'res_gmkcf_aio', 'kernel_list');
end

end