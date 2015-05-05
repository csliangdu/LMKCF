function LCCF_all_kernel(dataset, kernel_type, nRepeat)

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

res_dir = fullfile(pwd, [dataset, '_res']);
if ~exist(res_dir, 'dir')
    mkdir(res_dir);
end

disp(['Total number of Kernels: ', num2str(length(kernel_list)) '!']);

res_lccf_aio = [];
for iKernel = 1:length(kernel_list)
    iFile = kernel_list{iKernel};
    
    clear K;
    
    load(fullfile(kernel_dir, iFile), 'K');
    
	disp(['LCCF on ',  num2str(iKernel), ' of ' num2str(length(kernel_list)), ' Kernel(',  iFile(1:end-4), ') begin ...']);
    lccf_res_file = fullfile(res_dir, [iFile(1:end-4), '_res_lccf.mat']);
    if ~exist(lccf_res_file, 'file')
        t_start = clock;
        lccf_res = LCCF_single_kernel(K, y, fullfile(res_dir, iFile(1:end-4)), nRepeat);
        t_end = clock;
        disp(['LCCF exe time: ', num2str(etime(t_end, t_start))]);
        save(lccf_res_file, 'lccf_res');
    else
        load(lccf_res_file, 'lccf_res');
    end
    disp(['LCCF on ',  num2str(iKernel), ' of ' num2str(length(kernel_list)), ' Kernel(',  iFile(1:end-4), ') done']);
	if size(lccf_res, 1) > 1
		res_lccf_aio = [res_lccf_aio; mean(lccf_res)]; %#ok<AGROW>
	else
		res_lccf_aio = [res_lccf_aio; lccf_res]; %#ok<AGROW>
	end
    clear K lccf_res;
end

save(fullfile(res_dir, [dataset, '_res_lccf_all_kernel.mat']), 'res_lccf_aio', 'kernel_list');
end