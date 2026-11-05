%%%%%%%%%%%%%%%%%%%%%%%
% simulations shown in Fig. S4
% benchmarking MICs versus modulator splitting (a, b)
% testing MICs ability to identify a genuine modulator among a large pool of distractors (c, d)
%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%
% splitting simulation
%%%%%%%%%%%%%%%%%%%%%%%

% simulation settings
nS = 5; nY = 5; nXlist = [1, 3, 5, 7, 9]; % pop dimensionalities
N = 500; % num samples per sim
noise = 1; % emission noise
mod_stren = 1; % MIC strength
mA = 2; mC = 1; % AIS and MIC ranks
numSims = 100; % num sims per condition

% fitting parameters (MICs)
params = struct;
params.max_lin_rank = 10;
params.K = 10;
params.ctgs_folds_on = false; 
params.seed = 42; 
params.opts = struct;
params.max_mod_rank = 5;
params.stratify_by = [];

% fitting parameters (PC1 split)
Nsplits_l = 1:15;

% store results here
EV_mod_all = cell(numSims, length(nXlist)); % modulatory EV (MICs)
EV_split_mod_all = cell(numSims, length(nXlist)); % modulatory EV (PC1 split)

for nsns = 1:numSims
    for nxnx = 1:length(nXlist)
        rng("shuffle"); nX = nXlist(nxnx);

        % randomly oriented AIS and MIC
        UA = randn(nS, mA); UA = UA./vecnorm(UA, 2, 1);
        VA = randn(nY, mA); VA = VA./vecnorm(VA, 2, 1);
        A = UA*VA';
        WS = randn(nS, mC); WS = mod_stren*WS./vecnorm(WS, 2, 1);
        WY = randn(nY, mC); WY = WY./vecnorm(WY, 2, 1);
        WX = randn(nX, mC); WX = WX./vecnorm(WX, 2, 1);

        % simulate
        S = randn(N, nS);
        X = randn(N, nX);
        Y = S*A + ((S*WS).*(X*WX))*WY.' + noise*randn(N, nY);

        EV = nan(size(nXlist));
        EV_all = cell(size(EV));
        results = cell(size(EV));
        EV_A = nan(size(EV));
        EV_mod = nan(size(EV));

        % run PCA on modulator
        [coeff, score, latent, tsquared, explained, mu] = pca(zscore(X, [], 1));
        PC1 = score(:, 1); % PC1 score

        % use different numbers of splits of PC1 score
        for aaa = 1:length(Nsplits_l)
            disp([nsns, nxnx, aaa]);

            % identifier = index of split occupied by each sample
            identifier = nan(size(X, 1), 1);
            qq = linspace(0, 1, Nsplits_l(aaa)+1);
            for i = 1:Nsplits_l(aaa)
                identifier(quantile(PC1, qq(i))<=PC1 & PC1<=quantile(PC1, qq(i+1))) = i;
            end

            params.stratify_by = identifier; % stratify by the split index

            % run MICs with this stratification
            results{aaa} = MIC.model_wrapper(S, X, Y, params);
            EV_A(aaa) = results{aaa}.veA(results{aaa}.mA_try==results{aaa}.mA_opt);
            EV_mod(aaa) = results{aaa}.ve_mod(results{aaa}.mC_try==results{aaa}.mC_opt);

            % run PC1 split model with the same stratification
            EV_A_split = modulator_split(S, X, Y, identifier, params.K, Nsplits_l(aaa), params.seed);
            EV(aaa) = mean(EV_A_split);
            EV_all{aaa} = EV_A_split;
        end

        EV_mod_all{nsns, nxnx} = EV_mod;
        EV_split_mod_all{nsns, nxnx} = EV-EV_A;
    end
end

% plot results

figure;
subplot(2, 2, 1); hold on; % nX = 1, plot EV mod vs. # splits
abc = [];
def = [];
for i = 1:25
    abc(i, :) = EV_mod_all{i, 1};
    def(i, :) = EV_split_mod_all{i, 1};
end

errorbar(Nsplits_l, mean(abc, 1), std(abc, [], 1), 'LineWidth', 1.5);
errorbar(Nsplits_l, mean(def, 1), std(def, [], 1), 'LineWidth', 1.5);


peak_abc = nan(25, 5);
peak = nan(25, 5);
for j = 1:5
    for k = 1:25
        peak_abc(k, j) = max(EV_mod_all{k, j});
        peak(k, j) = max(EV_split_mod_all{k, j});
    end
end
subplot(2, 2, 2); hold on; % EV mod vs. nX, pick # splits with max EV mod
errorbar(nXlist, mean(peak_abc, 1), std(peak_abc, [], 1), 'LineWidth', 1.5);
errorbar(nXlist, mean(peak, 1), std(peak, [], 1), 'LineWidth', 1.5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% distractors simulation %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_distractors_list = [1 2 4 8 16]-1;
nS = 5; nY = 5; % population dimensionalities
N = 2000; % number of samples per sim
num_sims_per = 20;
mAgt = 2;
noise = 1;
mod_stren = 1;

% fitting parameters
params = struct;
params.K = 10;
params.ctgs_folds_on = false; 
params.seed = 42; 
params.opts = struct;
params.max_mod_rank = 5;
params.stratify_by = [];
params.max_lin_rank = 10;


saver = cell(1, num_sims_per);
for nn = 1:num_sims_per
    rng("shuffle");
    % randomly oriented AIS
    UAgt = randn(nS, mAgt); UAgt = UAgt./vecnorm(UAgt, 2, 1);
    VAgt = randn(nY, mAgt); VAgt = VAgt./vecnorm(VAgt, 2, 1);
    Agt = UAgt*VAgt';

    % randomly oriented MIC
    WSgt = randn(nS, 1); WSgt = mod_stren*WSgt./vecnorm(WSgt, 2, 1);
    WYgt = randn(nY, 1); WYgt = WYgt./vecnorm(WYgt, 2, 1);

    % simulate
    S = randn(N, nS);
    Xmax = randn(N, max(num_distractors_list)+1);
    noiseY = noise*randn(N, nY);

    abc = struct;
    abc.Agt = Agt;
    abc.WSgt = WSgt;
    abc.WYgt = WYgt;
    abc.S = S;
    abc.Xmax = Xmax;
    abc.noiseY = noiseY;

    results = cell(size(num_distractors_list));

    for ii = 1:length(num_distractors_list)
        disp([nn, ii])
        num_X_dims = num_distractors_list(ii) + 1;

        X = Xmax(:, 1:num_X_dims);
        WXgt = [1; zeros(num_X_dims-1, 1)];

        Y = S*Agt + ((S*WSgt).*(X*WXgt))*WYgt.' + noiseY;

        % fit
        out = MIC.model_wrapper(S, X, Y, params);
        results{ii} = out;
    end
    abc.results = results;
    saver{nn} = abc;
end

% plot results
ve_mod = [];
cos_angle = [];
rand_shuf_angle = [];
for i = 1:length(saver)
    for j = 1:length(num_distractors_list)
        curr = saver{i}.results{j};
        ve_mod(i, j) = curr.ve_mod(curr.mC_try==curr.mC_opt);
        for k = 1:params.K
            WX = curr.cv_mod{curr.mC_try==curr.mC_opt}.WX(:, :, k);
            cos_angle(i, j, k) = abs((WX/norm(WX))'*[1 ; zeros(length(WX)-1, 1)]);
        end

        if i == 1
            numshufs = 1e3;
            for nn = 1:numshufs
                WX = randn(size(WX));
                rand_shuf_angle(j, nn) = abs((WX/norm(WX))'*[1 ; zeros(length(WX)-1, 1)]);
            end
        end
    end
end

subplot(2, 2, 3);
errorbar(num_distractors_list, mean(ve_mod, 1), std(ve_mod, [], 1)/sqrt(size(ve_mod, 1)));

subplot(2, 2, 4);
errorbar(num_distractors_list, mean(rand_shuf_angle, 1), std(rand_shuf_angle, [], 1)/sqrt(size(rand_shuf_angle, 1)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% split pipeline %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function EV_A_split = modulator_split(S, X, Y, identifier, K, Nsplits, seed)

    maxEvals = 10; 
    
    % fit B and D without splitting
    mB = 0:min([size(X, 2), size(Y, 2)]); mD = 0:min([size(X, 2), size(S, 2)]);
    cvB = cell(size(mB)); cvD = cell(size(mD));
    veB = nan(size(cvB)); semB = nan(size(cvB)); veD = nan(size(cvD)); semD = nan(size(cvD));
    lambdaB = nan(size(cvB)); lambdaD = nan(size(cvD));
    
    for i = 1:length(mB)
        [cvB{i}, lambdaB(i)] = MIC.select_reg_RRR(K, false, identifier, seed, maxEvals, X, Y, mB(i));
        veB(i) = cvB{i}.mean_VE;
        semB(i) = std(cvB{i}.ve_fold)/sqrt(K);
    end
    
    for i = 1:length(mD)
        [cvD{i}, lambdaD(i)] = MIC.select_reg_RRR(K, false, identifier, seed, maxEvals, X, S, mD(i));
        veD(i) = cvD{i}.mean_VE;
        semD(i) = std(cvD{i}.ve_fold)/sqrt(K);
    end
    
    % select optimal mB, mD
    opt_idxB = MIC.select_rank(veB, semB);
    opt_idxD = MIC.select_rank(veD, semD);
    
    opt_cvBD = struct;
    opt_cvBD.B = cvB{opt_idxB}.A; 
    opt_cvBD.D = cvD{opt_idxD}.A;
    opt_cvBD.cv_partition = cvB{opt_idxB}.cv_partition;

    % now fit A for each split
    mA = 0:min([size(S, 2) size(Y, 2)]);
    cvA_split = cell(1, Nsplits);
    for nn = 1:Nsplits
        cvA = cell(size(mA)); veA = nan(size(cvA)); semA = nan(size(cvA)); lambdaA = nan(size(cvA)); 
        for i = 1:length(mA)
            [cvA{i}, lambdaA(i)] = select_reg_A_custom(S, X, Y, mA(i), opt_cvBD, maxEvals, identifier==nn);
            veA(i) = cvA{i}.mean_VE;
            semA(i) = std(cvA{i}.ve_fold)/sqrt(K);
        end
        opt_idxA = MIC.select_rank(veA, semA);
        cvA_split{nn} = cvA{opt_idxA};
    end

    % compute aggregate EV of split model
    EV_A_split = nan(1, K);
    rng(seed); cv = opt_cvBD.cv_partition;
    for k = 1:K
        tr = training(cv, k);     
        te = test(cv, k);
        [Strain, Stest] = MIC.split_normalize(S, tr, te);
        [Xtrain, Xtest] = MIC.split_normalize(X, tr, te);
        [Ytrain, Ytest] = MIC.split_normalize(Y, tr, te);

        % aggregate prediction
        identifier_test = identifier(te);
        Ypred = nan(size(Ytest));
        for nn = 1:Nsplits
            Ypred(identifier_test==nn, :) = Xtest(identifier_test==nn, :)*opt_cvBD.B(:, :, k) + (Stest(identifier_test==nn, :)-Xtest(identifier_test==nn, :)*opt_cvBD.D(:, :, k))*cvA_split{nn}.A(:, :, k);
        end

        % (B, D) prediction
        Ypred_BD = Xtest*opt_cvBD.B(:, :, k);
        
        % additional explained variance from A
        EV_A_split(k) = MIC.compute_EV(Ytrain, Ytest, Ypred) - MIC.compute_EV(Ytrain, Ytest, Ypred_BD);
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% helpers %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [cv_best, lambdaA] = select_reg_A_custom(S, X, Y, mA, cvBD, maxEvals, keep)

% choose range of regularization penalties to try
SS = zscore(S, [], 1); XX = zscore(X, [], 1);
SS = SS-XX*mean(cvBD.D, 3);

lambdasA = MIC.select_penalty_range(SS);

% Bayesian optimization of regularization penalties
% search space
v_lambdaA  = optimizableVariable('lambdaA', lambdasA, 'Transform','log');

if mA == 0
    disp("Warning: zero-rank A model (select_reg_A).");
    lambdaA = 0;
    cv_best = cv_fit_A_custom(cvBD, S, X, Y, mA, lambdaA, keep);
    return;
else
    objFcn = @(T) -cv_fit_A_custom(cvBD, S, X, Y, mA, T.lambdaA, keep).mean_VE;
end


opt_results = bayesopt(objFcn, v_lambdaA, ...
        'IsObjectiveDeterministic', true, ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'MaxObjectiveEvaluations', maxEvals, ...
        'UseParallel', false, 'Verbose', 0, 'PlotFcn', []); % remove 'PlotFcn', [] to show opt plots

lambdaA = opt_results.XAtMinObjective.lambdaA;

% refit best model
cv_best = cv_fit_A_custom(cvBD, S, X, Y, mA, lambdaA, keep);

% check whether selected lambdas are at boundary of search space
if log(lambdaA/lambdasA(1))<0.1
    disp("Warning: bumping up against lambdaA lower boundary.")
elseif log((lambdasA(end)/lambdaA))<0.1
    disp("Warning: bumping up against lambdaA upper boundary.")
end

end


function cv = cv_fit_A_custom(cvBD, S, X, Y, mA, lambdaA, keep)

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
        keep_train = keep(tr); keep_test = keep(te);
        
        Stest = Stest(keep_test, :); Xtest = Xtest(keep_test, :); Ytest = Ytest(keep_test, :);
        Strain = Strain(keep_train, :); Xtrain = Xtrain(keep_train, :); Ytrain = Ytrain(keep_train, :);

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
