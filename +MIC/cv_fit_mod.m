function cv = cv_fit_mod(cv_best_A, S, X, Y, mC, lambdaC, seed, opts) %+MIC

cvp = cv_best_A.cvBD.cv_partition;
K = cvp.NumTestSets;

nStemp = size(S, 2); nXtemp = size(X, 2); nYtemp = size(Y, 2);

if isfield(opts, "source_constraint")  
    if strcmp(opts.source_constraint, "same")
        nStemp = cv_best_A.mA;
    elseif strcmp(opts.source_constraint, "new")
        nStemp = nStemp - cv_best_A.mA;
    end
end

if isfield(opts, "target_constraint")  
    if strcmp(opts.target_constraint, "same")
        nYtemp = cv_best_A.mA;
    elseif strcmp(opts.target_constraint, "new")
        nYtemp = nYtemp - cv_best_A.mA;
    end
end

rank_upper_bound = max([nStemp*nYtemp nStemp*nXtemp nXtemp*nYtemp]);
if mC>rank_upper_bound
    mC = rank_upper_bound;
    disp("Warning: modulation rank exceeds max (cv_fit_mod).")
end

ve_fold = zeros(K, 1);
WSall = zeros(size(S, 2), mC, K);
WXall = zeros(size(X, 2), mC, K);
WYall = zeros(size(Y, 2), mC, K);

if mC>0
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

        if isfield(opts, "source_constraint") || isfield(opts, "target_constraint")
            [U, sigma, V] = svd(A); % U input, V output
            U = U(:, 1:cv_best_A.mA); V = V(:, 1:cv_best_A.mA);
        end

        % fit modulation term to residuals (possibly with constraints)
        src_sub = eye(size(A, 1)); trgt_sub = eye(size(A, 2));
        if isfield(opts, "source_constraint")  
            if strcmp(opts.source_constraint, "same")
                src_sub = U;
            elseif strcmp(opts.source_constraint, "new")
                src_sub = null(U');
            end
        end
        
        if isfield(opts, "target_constraint")  
            if strcmp(opts.target_constraint, "same")
                trgt_sub = V;
            elseif strcmp(opts.target_constraint, "new")
                trgt_sub = null(V');
            end
        end

        [WS_tilde, WX, WY_tilde] = MIC.fit_mod(Strain*src_sub, Xtrain, Yres_train*trgt_sub, mC, lambdaC, seed);
        WS = src_sub*WS_tilde; WY = trgt_sub*WY_tilde;

        % union + modulation prediction on test
        Ypred_test = Ypred_AB_test + ((Stest*WS).*(Xtest*WX))*WY.';

        % store model
        WSall(:, :, k) = WS;
        WXall(:, :, k) = WX;
        WYall(:, :, k) = WY;
        
        % modulation EV = full EV - linear model EV
        ve_fold(k) = MIC.compute_EV(Ytrain, Ytest, Ypred_test) - MIC.compute_EV(Ytrain, Ytest, Ypred_AB_test);
    end
else
    disp("Warning: zero-rank modulation model (cv_fit_mod).");
end

cv = struct();
cv.ve_fold = ve_fold;
cv.mean_VE = mean(ve_fold);
cv.cv_partition = cvp;
cv.WS = WSall;
cv.WX = WXall;
cv.WY = WYall;
cv.A_model = cv_best_A;
cv.lambdaC = lambdaC;
cv.seed = seed;

end