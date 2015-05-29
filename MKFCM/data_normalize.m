function data=data_normalize(data,method)
    if strcmp(method,'range')
        data=(data-repmat(min(data),size(data,1),1))./(repmat(max(data),size(data,1),1)-repmat(min(data),size(data,1),1));
    elseif strcmp(method,'var')
        data=(data-repmat(mean(data),size(data,1),1))./(repmat(std(data),size(data,1),1));
    else
        error('Unknown method given')
    end
