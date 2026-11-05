function [inner_model, EV_A, EV_B, EV_D, EV_mod, EV_A_shuf, EV_B_shuf, EV_D_shuf, EV_mod_shuf, p_A, p_B, p_D, p_mod] = ...
    heldout_test_cstr(S, X, Y, params, Kouter, nshuffle, shuffle_type) %+MIC
rng(params.seed);

% Permutation significance test for each component of the MICs model
% (Appendix A.10). Implements an outer Kouter-fold CV loop: on each fold,
% model_wrapper (Algorithm 1) is run on the training split to select ranks
% and penalties, the fitted model is then evaluated on the held-out test
% split. Significance is assessed by a residual-shuffle null: model
% predictions for each component are independently shuffled across test
% samples and the observed held-out EV is compared against the null
% distribution across nshuffle shuffles. One-tailed p-values are computed 
% empirically.
%
% NOTE ON VARIABLE NAMING: S, X, Y here correspond to X, Z, Y in the paper
% (source, modulator, target). See model_wrapper.m.
%
% Inputs:
%   S            [N x nS]   Source population (paper: X)
%   X            [N x nX]   Modulator (paper: Z)
%   Y            [N x nY]   Target population
%   params                  Parameter struct passed to model_wrapper;
%                           see model_wrapper.m for full field descriptions
%   Kouter       scalar     Number of outer CV folds for held-out evaluation.
%                           The inner CV fold count (for rank/penalty selection
%                           within each training split) is params.K.
%   nshuffle     scalar     Number of shuffle iterations for null distribution
%   shuffle_type string     Shuffle method:
%                           "circshift"  – circular shift by a random lag
%                                          drawn uniformly from [N/4, 3N/4];
%                                          preserves local temporal structure
%                           "randperm"   – independent random permutation of
%                                          test-set rows; destroys all structure
%
% Outputs:
%   inner_model  {1 x Kouter} cell   model_wrapper output structs, one per fold
%
%   EV_A         [1 x Kouter]   Held-out EV of baseline AIS (A), incremental
%                               over the modulator-only prediction (B)
%   EV_B         [1 x Kouter]   Held-out EV of additive Z->Y term (B)
%   EV_D         [1 x Kouter]   Held-out EV of additive Z->S term (D)
%   EV_mod       [1 x Kouter]   Held-out EVmod (C), incremental over A+B
%
%   EV_A_shuf    [nshuffle x Kouter]   Shuffle null for EV_A
%   EV_B_shuf    [nshuffle x Kouter]   Shuffle null for EV_B
%   EV_D_shuf    [nshuffle x Kouter]   Shuffle null for EV_D
%   EV_mod_shuf  [nshuffle x Kouter]   Shuffle null for EV_mod
%
%   p_A          scalar   One-tailed p-value for A (EV AIS)
%   p_B          scalar   One-tailed p-value for B
%   p_D          scalar   One-tailed p-value for D
%   p_mod        scalar   One-tailed p-value for C (EVmod)

inner_model = cell(1, Kouter);
EV_A = nan(1, Kouter); EV_mod = nan(1, Kouter); EV_B = nan(1, Kouter); EV_D = nan(1, Kouter);
EV_A_shuf = nan(nshuffle, Kouter); EV_mod_shuf = nan(nshuffle, Kouter); EV_B_shuf = nan(nshuffle, Kouter); EV_D_shuf = nan(nshuffle, Kouter);

% remove nans
keep = sum(isnan([S X Y]), 2)==0;
S = S(keep, :); X = X(keep, :); Y = Y(keep, :);

if params.ctgs_folds_on
    folds = kron(1:(Kouter-1), ones(1, floor(size(X, 1)/Kouter))); folds = [folds (folds(end)+1)*ones(1, size(X, 1)-length(folds))];
    cvp_outer = cvpartition('CustomPartition', folds);
elseif ~isempty(params.stratify_by)
    cvp_outer = cvpartition(params.stratify_by, 'KFold', Kouter);
else
    cvp_outer = cvpartition(size(X, 1), 'KFold', Kouter);
end

for whichOne = 1:Kouter
    %%%%%%%%%%%% split %%%%%%%%%%%%
    tr = training(cvp_outer, whichOne);
    te = test(cvp_outer, whichOne);

    Xtr = X(tr, :); Xte = X(te, :);
    Str = S(tr, :); Ste = S(te, :);
    Ytr = Y(tr, :); Yte = Y(te, :);

    % cross-val (internal to train)
    params2 = params;
    if ~isempty(params.stratify_by)
        params2.stratify_by = params.stratify_by(tr);
    end
    results_tr = MIC.model_wrapper(Str, Xtr, Ytr, params2);
    inner_model{whichOne} = results_tr;
    mA = results_tr.mA_opt;
    mB = results_tr.mB_opt;
    mC = results_tr.mC_opt;
    mD = results_tr.mD_opt;
    lambdaA = results_tr.lambdaA_opt;
    lambdaB = results_tr.lambdaB_opt;
    lambdaC = results_tr.lambdaC_opt;
    lambdaD = results_tr.lambdaD_opt;
    seed = params.seed;

    % standardize
    Ste = (Ste - mean(Str, 1))./std(Str, [], 1);
    Xte = (Xte - mean(Xtr, 1))./std(Xtr, [], 1);
    Yte = (Yte - mean(Ytr, 1))./std(Ytr, [], 1);

    Str = (Str - mean(Str, 1))./std(Str, [], 1);
    Xtr = (Xtr - mean(Xtr, 1))./std(Xtr, [], 1);
    Ytr = (Ytr - mean(Ytr, 1))./std(Ytr, [], 1);

    % fit B and D models to train data
    B = MIC.fit_RRR(Xtr, Ytr, mB, lambdaB);
    D = MIC.fit_RRR(Xtr, Str, mD, lambdaD);

    % fit A term
    A = MIC.fit_RRR(Str-Xtr*D, Ytr-Xtr*B, mA, lambdaA);

    % fit C term
    if strcmp(mC, "full")
        % form quadratic regressors
        Str_sub = Str - Xtr*D; Ste_sub = Ste - Xte*D;
        SXtr = nan(size(Str, 1), size(Str, 2)*size(Xtr, 2));
        for i = 1:size(SXtr, 1)
            curr = Str_sub(i, :)'*Xtr(i, :);
            SXtr(i, :) = curr(:)';
        end
        
        SXte = nan(size(Ste, 1), size(Ste, 2)*size(Xte, 2));
        for i = 1:size(SXte, 1)
            curr = Ste_sub(i, :)'*Xte(i, :);
            SXte(i, :) = curr(:)';
        end
        
        % fit
        Cfull = MIC.ridge_fit(SXtr, Ytr-Xtr*B-(Str-Xtr*D)*A, lambdaC);
    elseif mC>0
        % build constraint projectors from fitted A (mirrors cv_fit_mod)
        src_sub = eye(size(A,1)); trgt_sub = eye(size(A,2));
        if isfield(params.opts,"source_constraint") || isfield(params.opts,"target_constraint")
            [U,~,V] = svd(A); U = U(:,1:mA); V = V(:,1:mA);   % U input, V output
        end
        if isfield(params.opts,"source_constraint")
            if strcmp(params.opts.source_constraint,"same"), src_sub = U;
            elseif strcmp(params.opts.source_constraint,"new"), src_sub = null(U'); end
        end
        if isfield(params.opts,"target_constraint")
            if strcmp(params.opts.target_constraint,"same"), trgt_sub = V;
            elseif strcmp(params.opts.target_constraint,"new"), trgt_sub = null(V'); end
        end

        Sres = Str - Xtr*D;
        [WS_t, WX, WY_t] = MIC.fit_mod(Sres*src_sub, Xtr, (Ytr-Xtr*B-Sres*A)*trgt_sub, mC, lambdaC, seed);
        WS = src_sub*WS_t; WY = trgt_sub*WY_t;
    else
        WS = zeros(size(S, 2), 1); WX = zeros(size(X, 2), 1); WY = zeros(size(Y, 2), 1);
    end

    % test set performance of comm & mod models
    EV_D(whichOne) = MIC.compute_EV(Str, Ste, Xte*D);
    EV_B(whichOne) = MIC.compute_EV(Ytr, Yte, Xte*B);
    EV_A(whichOne) = MIC.compute_EV(Ytr, Yte, (Ste-Xte*D)*A + Xte*B) - MIC.compute_EV(Ytr, Yte, Xte*B);
    if strcmp(mC, "full")
        EV_mod(whichOne) = MIC.compute_EV(Ytr, Yte, (Ste-Xte*D)*A + Xte*B + SXte*Cfull) - MIC.compute_EV(Ytr, Yte, (Ste-Xte*D)*A + Xte*B);
    else
        EV_mod(whichOne) = MIC.compute_EV(Ytr, Yte, (Ste-Xte*D)*A + Xte*B + (((Ste-Xte*D)*WS).*(Xte*WX))*WY.') - MIC.compute_EV(Ytr, Yte, (Ste-Xte*D)*A + Xte*B);
    end

    % shuffle
    rng("shuffle");
    Dterm = Xte*D;
    Bterm = Xte*B;
    Aterm = (Ste-Xte*D)*A;
    if strcmp(mC, "full")
        Cterm = SXte*Cfull;
    else
        Cterm = (((Ste-Xte*D)*WS).*(Xte*WX))*WY.';
    end

    for nn = 1:nshuffle
        if strcmp(shuffle_type, "circshift")
            shifted_A = circshift(Aterm, randi(round([1/4 3/4]*size(Ste, 1))), 1);
            shifted_B = circshift(Bterm, randi(round([1/4 3/4]*size(Ste, 1))), 1);
            shifted_D = circshift(Dterm, randi(round([1/4 3/4]*size(Ste, 1))), 1);
            shifted_mod = circshift(Cterm, randi(round([1/4 3/4]*size(Ste, 1))), 1);
        else
            shifted_A = Aterm(randperm(size(Aterm, 1)), :);
            shifted_B = Bterm(randperm(size(Bterm, 1)), :);
            shifted_D = Dterm(randperm(size(Dterm, 1)), :);
            shifted_mod = Cterm(randperm(size(Cterm, 1)), :);
        end
        EV_B_shuf(nn, whichOne) = MIC.compute_EV(Ytr, Yte, shifted_B);
        EV_D_shuf(nn, whichOne) = MIC.compute_EV(Str, Ste, shifted_D);
        EV_A_shuf(nn, whichOne) = MIC.compute_EV(Ytr, Yte, shifted_A + Bterm) - MIC.compute_EV(Ytr, Yte, Bterm);
        EV_mod_shuf(nn, whichOne) = MIC.compute_EV(Ytr, Yte, Aterm + Bterm + shifted_mod) - MIC.compute_EV(Ytr, Yte, Aterm + Bterm);
    end

end


p_A = (1+sum(mean(EV_A_shuf, 2)>=mean(EV_A)))/(1+nshuffle);
p_B = (1+sum(mean(EV_B_shuf, 2)>=mean(EV_B)))/(1+nshuffle);
p_D = (1+sum(mean(EV_D_shuf, 2)>=mean(EV_D)))/(1+nshuffle);
p_mod = (1+sum(mean(EV_mod_shuf, 2)>=mean(EV_mod)))/(1+nshuffle);


end

