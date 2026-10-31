function cv = cv_fit_A(cvBD, S, X, Y, mA, lambdaA) %+MIC

% cvBD is cross-val for B, D terms (B: X-to-Y, D: X-to-S)
cvp = cvBD.cv_partition;
K = cvp.NumTestSets;

ve_fold = zeros(1, K);
Aall = zeros(size(S, 2), size(Y, 2), K);

if mA>0
    parfor k = 1:K
        tr = training(cvp,k);     
        te = test(cvp,k);
    
        [Strain, Stest] = MIC.split_normalize(S, tr, te);
        [Xtrain, Xtest] = MIC.split_normalize(X, tr, te);
        [Ytrain, Ytest] = MIC.split_normalize(Y, tr, te);
        

        B = cvBD.B(:, :, k);
        D = cvBD.D(:, :, k);

        Strain = Strain - Xtrain*D;
        Stest = Stest - Xtest*D;

        % Fit on training
        A = MIC.fit_RRR(Strain, Ytrain-Xtrain*B, mA, lambdaA);

        % Predict on test
        Ypred = Stest*A + Xtest*B;
    
        % store model
        Aall(:, :, k) = A;
    
        % EV of A = EV of (A, B) - EV of B
        ve_fold(k) = MIC.compute_EV(Ytrain, Ytest, Ypred) - MIC.compute_EV(Ytrain, Ytest, Xtest*B);
    end
else
    disp("Warning: zero-rank A-model (cv_fit_A).");
end

cv = struct();
cv.ve_fold = ve_fold;
cv.mean_VE = mean(ve_fold, 'omitnan');
cv.cvBD = cvBD;
cv.A = Aall;
cv.lambdaA = lambdaA;
cv.mA = mA;

end
