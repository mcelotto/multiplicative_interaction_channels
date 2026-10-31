function cv = cv_fit_mod_full_rank(cv_best_A, S, X, Y, lambdaC, seed) %+MIC

cvp = cv_best_A.cvBD.cv_partition;
K = cvp.NumTestSets;

ve_fold = zeros(K, 1);
Call = cell(1, K);

parfor k = 1:K
    tr = training(cvp,k);    
    te = test(cvp,k);

    [Strain, Stest] = MIC.split_normalize(S, tr, te);
    [Xtrain, Xtest] = MIC.split_normalize(X, tr, te);
    [Ytrain, Ytest] = MIC.split_normalize(Y, tr, te);
    
    A = cv_best_A.A(:, :, k);
    B = cv_best_A.cvBD.B(:, :, k);
    D = cv_best_A.cvBD.D(:, :, k);

    % residualize S
    Strain = Strain - Xtrain*D;
    Stest = Stest - Xtest*D;

    % linear prediction of Y
    Ypred_AB_train = Strain*A + Xtrain*B;
    Ypred_AB_test = Stest*A + Xtest*B;

    % residuals of linear prediction
    Yres_train = Ytrain - Ypred_AB_train;

    % form quadratic regressors
    SXtrain = nan(size(Strain, 1), size(Strain, 2)*size(Xtrain, 2));
    for i = 1:size(SXtrain, 1)
        curr = Strain(i, :)'*Xtrain(i, :);
        SXtrain(i, :) = curr(:)';
    end
    
    SXtest = nan(size(Stest, 1), size(Stest, 2)*size(Xtest, 2));
    for i = 1:size(SXtest, 1)
        curr = Stest(i, :)'*Xtest(i, :);
        SXtest(i, :) = curr(:)';
    end
    
    % fit
    Cfull = MIC.ridge_fit(SXtrain, Yres_train, lambdaC);
    
    % store in 3D form
    nS = size(Stest, 2);
    nX = size(Xtest, 2);
    nY = size(Ytest, 2);
    
    C3 = reshape(Cfull, [nS, nX, nY]);
    C3 = permute(C3, [3, 1, 2]);
   
    % store model
    Call{k} = C3;


    % union + modulation prediction on test
    Ypred_test = Ypred_AB_test + SXtest*Cfull;

    
    % modulation EV = full EV - linear model EV
    ve_fold(k) = MIC.compute_EV(Ytrain, Ytest, Ypred_test) - MIC.compute_EV(Ytrain, Ytest, Ypred_AB_test);
end


cv = struct();
cv.ve_fold = ve_fold;
cv.mean_VE = mean(ve_fold);
cv.cv_partition = cvp;
cv.C = Call;
cv.A_model = cv_best_A;
cv.lambdaC = lambdaC;
cv.seed = seed;

end
