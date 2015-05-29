function [res_lmkfcm, res_lmkfcm_detail] = LMKFCM_multi_kernel(Ks, cluster, gnd, m_degree_candi, nRepeat)

res_lmkfcm_detail = cell(length(m_degree_candi), 1);

for i1 = 1:length(m_degree_candi)
    
    m_degree = m_degree_candi(i1);
    
    rng('default');
    for i2 = 1:nRepeat
        [U, W] = GMKFCM(Ks, cluster, m_degree, struct('maxIter', [], 'nRepeat', 1), []);
        [~, idx] = max(U, [], 2);
        res_lmkfcm_detail{i1} = [res_lmkfcm_detail{i1}; ClusteringMeasure(gnd, idx)];
    end
    
end

res_lmkfcm.avg_res = cell2mat(cellfun(@(x) mean(x, 1), res_lmkfcm_detail, 'UniformOutput', 0));
res_lmkfcm.std_res = cell2mat(cellfun(@(x) std(x, 1, 1), res_lmkfcm_detail, 'UniformOutput', 0));
[~, best_m_id] = max(sum(res_lmkfcm.avg_res,2));
res_lmkfcm.m_degree = m_degree_candi(best_m_id);
res_lmkfcm.best_res = res_lmkfcm.avg_res(best_m_id, :);
end