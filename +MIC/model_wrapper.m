function results = model_wrapper(S, X, Y, params) % +MIC

% Fits the full MICs model (Eq. 5) via the hierarchical cross-validated
% procedure described in Algorithm 1. The model decomposes target variance
% into three components: additive modulator-to-target (B), baseline
% interaction subspace (A), and multiplicative interaction channels (C).
% An auxiliary regression of the modulator out of the source (D) is fit
% concurrently with B to residualize the source prior to A and C fitting;
% its cross-validated explained variance in the source is also reported.
% All components are rank-selected sequentially via cross-validated
% variance explained using a 1-SE stopping rule.
%
% NOTE ON VARIABLE NAMING: the paper uses X (source), Z (modulator), Y
% (target). This function uses S (source), X (modulator), Y (target).
%
% Inputs:
%   S          [N x nS]  Source population (paper: X)
%   X          [N x nX]  Modulator (paper: Z)
%   Y          [N x nY]  Target population
%   params     Struct with fields:
%     seed              Random seed for reproducibility
%     K                 Number of cross-validation folds
%     ctgs_folds_on     If true, use temporally contiguous CV folds
%     stratify_by       [1 x N] variable used for fold stratification;
%                       not compatible with ctgs_folds_on
%     max_lin_rank      Maximum rank for A, B, and D searches
%     max_mod_rank      Maximum rank mC for the modulation tensor C
%     opts              Struct for geometry-constrained fitting (optional):
%       opts.target_constraint  "same" (wY in AIS) | "new" (wY out of AIS)
%       opts.source_constraint  "same" (wS in AIS) | "new" (wS out of AIS)
%                               Omit field for unconstrained fitting
%     mA, mB, mC, mD   Override rank search with fixed rank vector (optional)
%
% Outputs:
%   mA_opt, mB_opt, mC_opt, mD_opt        Selected ranks for A, B, C, D
%   lambdaA_opt, lambdaB_opt,
%     lambdaC_opt, lambdaD_opt            Selected regularization penalties
%   mA_try, mB_try, mC_try, mD_try        Full rank grids searched
%   nS, nX, nY                            Input dimensionalities
%
%   cvB    [1 x |mB|] cell  CV result structs for each B rank; cvB{i}.A
%                           contains the fitted B matrix for rank mB(i)
%   veB    [1 x |mB|]       Mean cross-validated VE of Z->Y (B), per rank
%   semB   [1 x |mB|]       Across-fold SEM of veB
%
%   cvD    [1 x |mD|] cell  CV result structs for each D rank
%   veD    [1 x |mD|]       Mean cross-validated VE of Z->S (D), per rank
%   semD   [1 x |mD|]       Across-fold SEM of veD
%
%   cvA    [1 x |mA|] cell  CV result structs for each A rank
%   veA    [1 x |mA|]       Mean cross-validated VE of baseline AIS (A),
%                           per rank (incremental over B alone)
%   semA   [1 x |mA|]       Across-fold SEM of veA
%
%   cv_mod [1 x |mC|] cell  CV result structs for each modulation rank
%   ve_mod [1 x |mC|]       Mean cross-validated EVmod (C), per rank
%                           (incremental over additive model A+B)
%   sem_mod [1 x |mC|]      Across-fold SEM of ve_mod


%%%%%%%%%%%%%%%%%%%%%%%
maxEvals = 10; % number of evals of bayesopt in penalty searches
%%%%%%%%%%%%%%%%%%%%%%%

% store results
results = struct;

% remove nans
keep = sum(isnan([S X Y]), 2)==0;
S = S(keep, :); X = X(keep, :); Y = Y(keep, :);

% exhaustive rank search
mB = 0:min([size(X, 2), size(Y, 2), params.max_lin_rank]);
mD = 0:min([size(X, 2), size(S, 2), params.max_lin_rank]);
mA = 0:min([size(S, 2) size(Y, 2), params.max_lin_rank]);

if isfield(params, "mA")
    mA = params.mA;
end

if isfield(params, "mB")
    mB = params.mB;
end

if isfield(params, "mD")
    mD = params.mD;
end

cvB = cell(size(mB));
cvD = cell(size(mD));
cvA = cell(size(mA));

veB = nan(size(cvB));
semB = nan(size(cvB));
veD = nan(size(cvD));
semD = nan(size(cvD));
veA = nan(size(cvA));
semA = nan(size(cvA));

lambdaB = nan(size(cvB)); 
lambdaD = nan(size(cvD));
lambdaA = nan(size(cvA)); 

for i = 1:length(mB)
    [cvB{i}, lambdaB(i)] = MIC.select_reg_RRR(params.K, params.ctgs_folds_on, params.stratify_by, params.seed, maxEvals, X, Y, mB(i));
    veB(i) = cvB{i}.mean_VE;
    semB(i) = std(cvB{i}.ve_fold)/sqrt(params.K);
end

for i = 1:length(mD)
    [cvD{i}, lambdaD(i)] = MIC.select_reg_RRR(params.K, params.ctgs_folds_on, params.stratify_by, params.seed, maxEvals, X, S, mD(i));
    veD(i) = cvD{i}.mean_VE;
    semD(i) = std(cvD{i}.ve_fold)/sqrt(params.K);
end

% select B and D ranks
opt_idxB = MIC.select_rank(veB, semB);
opt_idxD = MIC.select_rank(veD, semD);

opt_cvBD = struct;
opt_cvBD.B = cvB{opt_idxB}.A; 
opt_cvBD.D = cvD{opt_idxD}.A;
opt_cvBD.cv_partition = cvB{opt_idxB}.cv_partition;

for i = 1:length(mA)
    [cvA{i}, lambdaA(i)] = MIC.select_reg_A(S, X, Y, mA(i), opt_cvBD, maxEvals);
    veA(i) = cvA{i}.mean_VE;
    semA(i) = std(cvA{i}.ve_fold)/sqrt(params.K);
end

% select A rank
opt_idxA = MIC.select_rank(veA, semA);

% modulation term
mC = 0:min([params.max_mod_rank, size(S, 2)*size(Y, 2) size(X, 2)*size(Y, 2) size(S, 2)*size(X, 2)]);

if isfield(params, "mC")
    mC = params.mC;
end

cv_mod = cell(size(mC));
ve_mod = nan(size(mC));
sem_mod = nan(size(mC));
lambdaC = nan(size(mC));

if isfield(params, "als_seed")
    als_seed = params.als_seed;
else
    als_seed = params.seed;
end

for ll = 1:length(mC) 
    disp(['modulation model...' num2str(ll)]);
    [cv_mod{ll}, lambdaC(ll)] = MIC.select_reg_mod(cvA{opt_idxA}, S, X, Y, mC(ll), als_seed, maxEvals, params.opts);
    ve_mod(ll) = cv_mod{ll}.mean_VE;
    sem_mod(ll) = std(cv_mod{ll}.ve_fold)/sqrt(params.K);
end

% select modulation rank
opt_idxC = MIC.select_rank(ve_mod, sem_mod);

% store results
results.mA_opt = mA(opt_idxA);
results.mB_opt = mB(opt_idxB);
results.mC_opt = mC(opt_idxC);
results.mD_opt = mD(opt_idxD);

results.lambdaA_opt = lambdaA(opt_idxA);
results.lambdaB_opt = lambdaB(opt_idxB);
results.lambdaC_opt = lambdaC(opt_idxC);
results.lambdaD_opt = lambdaD(opt_idxD);

results.mA_try = mA; 
results.mB_try = mB; 
results.mC_try = mC;
results.mD_try = mD;

results.nS = size(S, 2);
results.nX = size(X, 2);
results.nY = size(Y, 2);

results.cvB = cvB;
results.veB = veB;
results.semB = semB;

results.cvD = cvD;
results.veD = veD;
results.semD = semD;

results.cvA = cvA;
results.veA = veA;
results.semA = semA;

results.cv_mod = cv_mod;
results.ve_mod = ve_mod;
results.sem_mod = sem_mod;

end