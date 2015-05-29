function [res_gmkfcm, res_gmkfcm_detail] = GMKFCM_multi_kernel(Ks, cluster, gnd, m_degree_candi, nRepeat)

res_gmkfcm_detail = cell(length(m_degree_candi), 1);

for i1 = 1:length(m_degree_candi)
    
    m_degree = m_degree_candi(i1);
    
    rng('default');
    for i2 = 1:nRepeat
        [U, W] = LMKFCM(Ks, cluster, m_degree, struct('maxIter', [], 'nRepeat', 1), []);
        [~, idx] = max(U, [], 2);
        res_gmkfcm_detail{i1} = [res_gmkfcm_detail{i1}; ClusteringMeasure(gnd, idx)];
    end
    
end

res_gmkfcm.avg_res = cell2mat(cellfun(@(x) mean(x, 1), res_gmkfcm_detail, 'UniformOutput', 0));
res_gmkfcm.std_res = cell2mat(cellfun(@(x) std(x, 1, 1), res_gmkfcm_detail, 'UniformOutput', 0));
[~, best_m_id] = max(sum(res_gmkfcm.avg_res,2));
res_gmkfcm.m_degree = m_degree_candi(best_m_id);
res_gmkfcm.best_res = res_gmkfcm.avg_res(best_m_id, :);
end