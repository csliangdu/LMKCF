function res_aio = run_all_algos(inXcell, Ks, y, algo_list)

nView = length(inXcell);
nKernel = size(Ks, 3);
nReplicate = 1;
for iAlgo = 1:length(algo_list)
    switch lower(algo_list{iAlgo})
        case lower('nmf')
            res_nmf = cell(nView, nRepeat);
            for i1 = 1:nView
                for i2 = 1:nRepeat
                    t_start = clock;
                    [~, V, objHistory] = nmf(inXcell{i1}, nCluster, struct('replicates', nReplicate));
                    t_end = clock;
                    label_nmf = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 20);
                    res_nmf{i1, i2} = struct('res', ClusteringMeasure(y, label_nmf), 'obj', objHistory(end), 'time', etime(t_end, t_start));
                end
            end
        case lower('km')
            res_km = cell(nView, nRepeat);
            for i1 = 1:nView
                for i2 = 1:nRepeat
                    t_start = clock;
                    [label_km, ~, ~, objHistory] = litekmeans(inXcell{i1}, nClass, 'maxIter', 1000, 'Replicates', nReplicate);
                    t_end = clock;
                    res_km{i1, i2} = struct('res', ClusteringMeasure(y, label_km), 'obj', objHistory(end), 'time', etime(t_end, t_start));
                end
            end
        case lower('kkm')
            res_kkm = cell(nKernel, nRepeat);
            for i1 = 1:nKernel
                for i2 = 1:nRepeat
                    t_start = clock;
                    label_kkm = KernelKmeans(Ks(:,:,i1), nClass, 'maxiter', 1000, 'Replicates', nReplicate);
                    t_end = clock;
                    res_kkm{i1, i2} = struct('res', ClusteringMeasure(y, label_kkm), 'obj', objHistory(end), 'time', etime(t_end, t_start));
                end
            end
        case lower('sc')
            res_sc = cell(nKernel, nRepeat);
            for i1 = 1:nKernel
                t_start = clock;
                V = sc_on_kernel(Ks(:,:,i1), nClass);
                t_end = clock;
                for i2 = 1:nRepeat
                    label_sc = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 20);
                    res_sc{i1, i2} = struct('res', ClusteringMeasure(y, label_sc), 'obj', 0, 'time', etime(t_end, t_start));
                end
            end
        case lower('kcf')
            res_kcf = cell(nKernel, nRepeat);
            for i1 = 1:nKernel
                for i2 = 1:nRepeat
                    t_start = clock;
                    [~, V, ~, objHistory] = KCF_Multi(Ks(:,:,i1), nClass, struct('maxIter', [], 'nRepeat', nReplicate), [], []);
                    t_end = clock;
                    label_kcf = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 20);
                    res_kcf{i1, i2} = struct('res', ClusteringMeasure(y, label_kcf), 'obj', objHistory(end), 'time', etime(t_end, t_start));
                end
            end
        case lower('lccf')
            res_lccf = cell(nKernel, nRepeat);
            for i1 = 1:nKernel
                for i2 = 1:nRepeat
                    t_start = clock;
                    [~, V, ~, objHistory] = LCCF_simple(Ks(:,:,i1), nClass, 5, struct('maxIter', [], 'nRepeat', nReplicate, 'alpha', 100), [], []);
                    t_end = clock;
                    label_lccf = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 20);
                    res_lccf{i1, i2} = struct('res', ClusteringMeasure(y, label_lccf), 'obj', objHistory(end), 'time', etime(t_end, t_start));
                end
            end
        case lower('rmvkm')
            gamma_values = 0.7;
            nParam = length(gamma_values);
            res_rmvkm = cell(nParam, nRepeat);
            for i1 = 1:nParam
                for i2 = 1:nRepeat
                    t_start = clock;
                    
                    param = [];
                    param.r = gamma_values(i1);
                    param.numCluster = nClass;
                    param.maxIter = 100;
                    param.thresh = 1e-3;
                    
                    ridx = randi(nClass, nSmp, 1);
                    inG0 = full(sparse((1:nSmp)', ridx, ones(nSmp,1), nSmp, nClass));
                    [outG0, ~, ~, ~, objHistory] = weighted_robust_multi_kmeans( inXCell, param, inG0 );
                    [~, label_rmvkm, ~] = find(outG0);
                    
                    t_end = clock;
                    res_rmvkm{i1, i2} = struct('res', ClusteringMeasure(y, label_rmvkm), 'obj', objHistory(end), 'time', etime(t_end, t_start));
                end
            end
        case lower('mvnmf')
            alpha_values = 0.01;
            nParam = length(alpha_values);
            res_mvnmf = cell(nParam, nRepeat);
            inXcell_norm = inXcell;
            for i1 = 1:nView
                inXcell_norm{i1} = inXcell_norm{i1} / sum(sum(inXcell_norm{i1}));
            end
            for i1 = 1:nParam
                for i2 = 1:nRepeat
                    t_start = clock;
                    options = [];
                    options.maxIter = 200;
                    options.error = 1e-6;
                    options.nRepeat = 30;
                    options.minIter = 50;
                    options.meanFitRatio = 0.1;
                    options.rounds = 30;
                    options.alpha = ones(1,nView)*alpha_values(i1);
                    options.kmeans = 0;
                    [~, ~, V, objHistory] = MultiNMF(inXcell_norm, nClass, y, options);
                    t_end = clock;
                    label_mvnmf = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 20);
                    res_mvnmf{i1, i2} = struct('res', ClusteringMeasure(y, label_mvnmf), 'obj', objHistory(end), 'time', etime(t_end, t_start));
                end
            end
            clear inXcell_norm;
        case lower('mkkm')
            res_mkkm = cell(1, nRepeat);
            for i1 = 1:nRepeat
                t_start = clock;
                U = rand(nSmp, nClass);
                [~, uidx] = max(U, [], 2);
                U = zeros(size(U));
                U(sub2ind(size(U), (1:size(U,1))', uidx)) = 1;
                [label_mkkm, ~, objHistory] = MKKM(U, nClass, 1, 1e-5, Ks);
                t_end = clock;
                res_mkkm{i1} = struct('res', ClusteringMeasure(y, label_mkkm), 'obj', objHistory(end), 'time', etime(t_end, t_start));
            end
        case lower('gmkkm')
            res_gmkkm = cell(1, nRepeat);
            for i1 = 1:nRepeat
                t_start = clock;
                state = mkkmeans_train(Ks, struct('cluster_count', nClass, 'iteration_count', 10));
                t_end = clock;
                res_gmkkm{i1} = struct('res', ClusteringMeasure(y, state.clustering), 'obj', state.objective(end), 'time', etime(t_end, t_start));
            end
        case lower('lmkkm')
            res_lmkkm = cell(1, nRepeat);
            for i1 = 1:nRepeat
                t_start = clock;
                state = lmkkmeans_train(Ks, struct('cluster_count', nClass, 'iteration_count', 10));
                t_end = clock;
                res_lmkkm{i1} = struct('res', ClusteringMeasure(y, state.clustering), 'obj', state.objective(end), 'time', etime(t_end, t_start));
            end
        case lower('rmkkm')
            gamma_values = 0.4;
            nParam = length(gamma_values);
            res_rmkkm = cell(nParam, nRepeat);
            for i1 = 1:nParam
                gamma = gamma_values(i1);
                for i2 = 1:nRepeat
                    t_start = clock;
                    [label_rmkkm, ~, ~, ~, objHistory] = RMKKM(Ks, nClass, 'gamma', gamma, 'maxiter', 50, 'replicates', 1);
                    t_end = clock;
                    res_rmkkm{i1, i2} = struct('res', ClusteringMeasure(y, label_rmkkm), 'obj', objHistory(end), 'time', etime(t_end, t_start));
                end
            end
        case lower('coreg_centroid')
            lambda_values = 1;
            nParam = length(lambda_values);
            res_coreg_centriod = cell(nParam, nRepeat);
            for i1 = 1:nParam
                t_start = clock;
                opts.lambda = lambda_values(i1);
                opts.maxiter = 50;
                V = coreg_centroid_on_multi_kernel(Ks, nClass, opts);
                t_end = clock;
                for i2 = 1:nRepeat
                    label_coreg_centriod = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 20);
                    res_coreg_centriod{i1, i2} = struct('res', ClusteringMeasure(y, label_coreg_centriod), 'obj', 0, 'time', etime(t_end, t_start));
                end
            end
        case lower('rmsc')
            lambda_values = 0.005;
            nParam = length(lambda_values);
            res_rmsc = cell(nParam, nRepeat);
            for i1 = 1:nParam
                t_start = clock;
                lambda = lambda_values(i1);
                opts.DEBUG = 0;
                %opts.mu = 1e-6;
                %opts.rho = 1.2;
                opts.eps = 1e-6;
                opts.max_iter = 300;
                P_hat = RMSC(Ks, lambda, opts);
                V = sc_on_kernel(P_hat, nClass);
                t_end = clock;
                for i2 = 1:nRepeat
                    label_rmsc = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 20);
                    res_rmsc{i1, i2} = struct('res', ClusteringMeasure(y, label_rmsc), 'obj', 0, 'time', etime(t_end, t_start));
                end
            end
        case lower('gmkcf')
            res_gmkcf = cell(1, nRepeat);
            for i1 = 1:nRepeat
                t_start = clock;
                [~, V, ~, ~, objhistory_final] = LMKCF_Multi(Ks, nClass, struct('maxIter', [], 'nRepeat', nReplicate), [], []);
                t_end = clock;
                label_gmkcf = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 20);
                res_gmkcf{i1} = struct('res', ClusteringMeasure(y, label_gmkcf), 'obj', objhistory_final(end), 'time', etime(t_end, t_start));
            end
        case lower('lmkcf')
            res_lmkcf = cell(1, nRepeat);
            for i1 = 1:nRepeat
                t_start = clock;
                [~, V, ~, ~, objhistory_final] = LMKCF_Multi(Ks, nClass, struct('maxIter', [], 'nRepeat', nReplicate), [], []);
                t_end = clock;
                label_lmkcf = litekmeans(V, nClass, 'maxIter', 1000, 'Replicates', 20);
                res_lmkcf{i1} = struct('res', ClusteringMeasure(y, label_lmkcf), 'obj', objhistory_final(end), 'time', etime(t_end, t_start));
            end
        otherwise
            error('Not supported yet!');
    end
end
end

function extract_result(res_cell, res_type)

if res_type == 1
    [nView, nRepeat] = size(res_cell);
elseif res_type == 2
    [nView, nRepeat] = size(res_cell);
elseif res_type == 3
    [nView, nRepeat] = size(res_cell);
elseif res_type == 4
    [nView, nRepeat] = size(res_cell);
end
end