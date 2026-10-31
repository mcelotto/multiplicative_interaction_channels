function cv = cv_fit_RRR(K, ctgs_folds_on, stratify_by, seed, X, Y, m, lambda)
rng(seed); % set random seed for reproducability

% Build K folds
if ctgs_folds_on
    folds = kron(1:(K-1), ones(1, floor(size(X, 1)/K))); folds = [folds (folds(end)+1)*ones(1, size(X, 1)-length(folds))];
    cvp = cvpartition('CustomPartition', folds);
elseif ~isempty(stratify_by)
    cvp = cvpartition(stratify_by, 'KFold', K);
else
    cvp = cvpartition(size(X, 1), 'KFold', K);
end

ve_fold = zeros(1,K);
Aall = zeros(size(X, 2), size(Y, 2), K);

if m>0
    parfor k = 1:K
        tr = training(cvp, k);     
        te = test(cvp, k);
    
        [Xtrain, Xtest] = MIC.split_normalize(X, tr, te);
        [Ytrain, Ytest] = MIC.split_normalize(Y, tr, te);
        
        % Fit on training
        A = MIC.fit_RRR(Xtrain, Ytrain, m, lambda);
    
        % Predict on test
        Ypred = Xtest*A;
    
        % store model
        Aall(:, :, k) = A;
    
        % EV on test
        ve_fold(k) = MIC.compute_EV(Ytrain, Ytest, Ypred);
    end
else
    disp("Warning: zero-rank union model (cv_fit_union).");
end

cv = struct();
cv.ve_fold = ve_fold;
cv.mean_VE = mean(ve_fold, 'omitnan');
cv.cv_partition = cvp;
cv.A = Aall;
cv.lambda = lambda;
cv.m = m;
cv.seed = seed;

end