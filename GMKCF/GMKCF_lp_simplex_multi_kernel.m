function GMKCF_lp_simplex_multi_kernel(dataset, kernel_type, nRepeat)
gammaCandidates = (0.1:0.1:0.9);

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
Ks = cell(nKernel,1);

for iKernel = 1:length(kernel_list)
    iFile = kernel_list{iKernel};
    
    clear K;
    
    load(fullfile(kernel_dir, iFile), 'K');
    Ks{iKernel} = K;
end
clear K;

disp(['Total number of Kernels: ', num2str(length(kernel_list)) '!']);

disp('GMKCF_lp_simplex on multi kernel begin ...');

res_gmkcflp_file = fullfile(res_dir, [dataset, '_res_gmkcflp_multi_kernel.mat']);

if exist(res_gmkcflp_file, 'file')
    load(res_gmkcflp_file, 'res_gmkcflp_aio', 'kernel_list', 'best_gamma', 'res_aio_p','best_res_gmkcflp_mean');
else
    res_aio_p = [];
    best_val = 0;
    for gammaIdx = 1:length(gammaCandidates)
        gamma = gammaCandidates(gammaIdx);
        
        res_gmkcflp = [];
        res_gmkcflp_obj = [];
        res_gmkcflp_mean = [];
        obj_final = [];
        kw_gmkcflp = cell(nRepeat, 1);
        
        res_file_gamma = fullfile(res_dir, strcat(dataset, '_res_gmkcflp_gamma=', num2str(gamma), '.mat'));
        if iscell(res_file_gamma); res_file_gamma = res_file_gamma{1}; end
        if exist(res_file_gamma, 'file')
            load(res_file_gamma, 'res_gmkcflp', 'obj_final', 'kw_gmkcflp');
        else
            rng('default');
            for iRepeat = 1:nRepeat
                t_start = clock;
                disp(['GMKCF_lp_simplex with gamma = ', num2str(gamma), ' ', num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations begin ...']);
                [~, V, kw, ~, objhistory_final] = GMKCF_Multi(Ks, nClass, gamma, struct('maxIter', 20, 'nRepeat', 1), [], []);
                label_gmkcflp = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 20);
                res_gmkcflp = [res_gmkcflp; ClusteringMeasure(y, label_gmkcflp)];%#ok<AGROW>
                obj_final = [obj_final; objhistory_final(end)];%#ok<AGROW>
                kw_gmkcflp{iRepeat} = kw;
                t_end = clock;
                disp(['GMKCF_lp_simplex with gamma = ', num2str(gamma), ' ', num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations done']);
                disp(['GMKCF_lp_simplex exe time: ', num2str(etime(t_end, t_start))]);
            end
            save(res_file_gamma, 'res_gmkcflp', 'obj_final', 'kw_gmkcflp', 'gamma');
        end
        if nRepeat > 1
            [~, minIdx] = min(obj_final);
            res_gmkcflp_obj = [res_gmkcflp_obj; res_gmkcflp(minIdx(1),:)];%#ok
            kw_obj = kw_gmkcflp{minIdx(1)};
            res_gmkcflp_mean = mean(res_gmkcflp);
        else
            res_gmkcflp_mean = [res_gmkcflp_mean; res_gmkcflp];%#ok
        end
        res_aio_p = [res_aio_p; res_gmkcflp_mean];%#ok
        
        if sum(res_gmkcflp_mean(:)) > best_val
            best_gamma = gamma;%#ok
            best_res_gmkcflp_mean = res_gmkcflp_mean;
            if nRepeat > 1
                best_kw_obj = kw_obj;%#ok
                best_res_gmkcflp_obj = res_gmkcflp_obj;%#ok
            end
        end
    end
    res_gmkcflp_aio = best_res_gmkcflp_mean;%#ok
   
    clear Ks K res_gmkcflp;
    
    save(res_gmkcflp_file, 'res_gmkcflp_aio', 'kernel_list', 'best_gamma', 'res_aio_p','best_res_gmkcflp_mean');
end

 disp('GMKCF_lp_simplex on multi kernel done');
end