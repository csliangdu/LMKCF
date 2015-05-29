function paramCell = buildParam_mkfc(p1Candidates)
if ~exist('p1Candidates', 'var')
    p1Candidates = [1.08];
end

n1 = length(p1Candidates);

nP = n1;

paramCell = cell(nP, 1);
idx = 0;

for i1 = 1:n1
    param = [];
    if ~isempty(p1Candidates)
        param.degree = p1Candidates(i1);
    end
    param.threshold=1e-5;
    
    idx = idx + 1;
    paramCell{idx} = param;
end
end