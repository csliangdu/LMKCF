function coreg_centroid_multi_kernel(dataset, kernel_type, nRepeat)
lambdaCandidates = 10.^(-2:2);

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
Ks = zeros(length(y),length(y),nKernel);

for iKernel = 1:length(kernel_list)
    iFile = kernel_list{iKernel};
    
    clear K;
    
    load(fullfile(kernel_dir, iFile), 'K');
    Ks(:,:,iKernel) = K;
end
clear K;

disp(['Total number of Kernels: ', num2str(length(kernel_list)) '!']);

disp('coreg_centroid on multi kernel begin ...');

res_coreg_centroid_file = fullfile(res_dir, [dataset, '_res_coreg_centroid_multi_kernel.mat']);

if exist(res_coreg_centroid_file, 'file')
    load(res_coreg_centroid_file, 'res_coreg_centroid_aio', 'kernel_list', 'best_lambda', 'res_aio_p','best_res_coreg_centroid_mean');
else
    res_aio_p = [];
    best_val = 0;
    for lambdaIdx = 1:length(lambdaCandidates)
        lambda = lambdaCandidates(lambdaIdx);
        
        res_coreg_centroid = [];
        res_coreg_centroid_obj = [];
        res_coreg_centroid_mean = [];
        obj_final = [];
        kw_coreg_centroid = cell(nRepeat, 1);
        
        res_file_lambda = fullfile(res_dir, strcat(dataset, '_res_coreg_centroid_lambda=', num2str(lambda), '.mat'));
        if iscell(res_file_lambda); res_file_lambda = res_file_lambda{1}; end
        if exist(res_file_lambda, 'file')
            load(res_file_lambda, 'res_coreg_centroid', 'obj_final', 'kw_coreg_centroid');
        else
            
            t_start = clock;
            opts = [];
            opts.lambda = lambda;
            opts.maxiter = 10;
            V = coreg_centroid_on_multi_kernel(Ks, nClass, opts);
            t_end = clock;
            
            rng('default');
            for iRepeat = 1:nRepeat
                disp(['coreg_centroid with lambda = ', num2str(lambda), ' ', num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations begin ...']);
                label_coreg_centriod = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 20);
                res_coreg_centroid = [res_coreg_centroid; ClusteringMeasure(y, label_coreg_centriod)];%#ok<AGROW>
                obj_final = [obj_final; 0];%#ok<AGROW>
                kw_coreg_centroid{iRepeat} = 0;
                disp(['coreg_centroid with lambda = ', num2str(lambda), ' ', num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations done']);
                disp(['coreg_centroid exe time: ', num2str(etime(t_end, t_start))]);
            end
            save(res_file_lambda, 'res_coreg_centroid', 'obj_final', 'kw_coreg_centroid', 'lambda');
        end
        if nRepeat > 1
            [~, minIdx] = min(obj_final);
            res_coreg_centroid_obj = [res_coreg_centroid_obj; res_coreg_centroid(minIdx(1),:)];%#ok
            kw_obj = kw_coreg_centroid{minIdx(1)};
            res_coreg_centroid_mean = mean(res_coreg_centroid);
        else
            res_coreg_centroid_mean = [res_coreg_centroid_mean; res_coreg_centroid];%#ok
        end
        res_aio_p = [res_aio_p; res_coreg_centroid_mean];%#ok
        
        if sum(res_coreg_centroid_mean(:)) > best_val
            best_lambda = lambda;%#ok
            best_res_coreg_centroid_mean = res_coreg_centroid_mean;
            if nRepeat > 1
                best_kw_obj = kw_obj;%#ok
                best_res_coreg_centroid_obj = res_coreg_centroid_obj;%#ok
            end
        end
    end
    res_coreg_centroid_aio = best_res_coreg_centroid_mean;%#ok
    
    clear Ks K res_coreg_centroid;
    
    save(res_coreg_centroid_file, 'res_coreg_centroid_aio', 'kernel_list', 'best_lambda', 'res_aio_p','best_res_coreg_centroid_mean');
end

disp('coreg_centroid on multi kernel done');
end