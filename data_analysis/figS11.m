%%%%%%%%%%%%%%%%%%%%%%%%%%
% code to produce Fig S11
% robustness of modulation EV and geometry indices 
% to non-convexity of modulation loss function (a, b)
% comparison to splitting-based pipeline (c-e)
% comparison to wheel-only and pupil-only fits (f)
% nonlinearity diagnostic (g)
%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%
% (a, b)
%%%%%%%%

% fitting parameters
params = struct;
params.max_lin_rank = 20;
params.K = 10;
params.ctgs_folds_on = true; 
params.seed = 42;
params.opts = struct;
params.stratify_by = [];
params.max_mod_rank = 10;

% previously selected ranks
mA = [2 2 2 2 3 2 1 1 2 1 1 1 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 1];
mB = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 0 1 1 1 1 1 1 1];
mC = [1 3 3 0 3 1 0 3 0 1 1 1 3 1 2 2 2 1 1 0 0 2 4 0 2 1 2 0];
mD = [1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1];

% number of choices of ALS fitting seed per session
num_seeds_per = 10;

res = cell(28, num_seeds_per);
for idx = 1:28
    % read data
    [ACA, VIP, wheel, pupil, stim_on, puff, t, type] = readACAVIP(idx);
    disp(size(VIP)); disp(size(ACA))
    X = [wheel pupil];
    S = ACA;
    Y = VIP;

    % pre-selected ranks
    params2 = params;
    params2.mA = mA(idx); params2.mB = mB(idx); params2.mC = mC(idx); params2.mD = mD(idx);

    for l = 1:num_seeds_per
        disp([idx, l]);
        rng("shuffle")
        params2.als_seed = randi(100000); % set ALS fitting seed
        res{idx, l} = MIC.model_wrapper(S, X, Y, params2);
    end
end

% aggregate results
ve_mod = nan(size(res));
alpha_sub = nan([size(res) 4]);
for i = 1:size(res, 1)
    for j = 1:size(res, 2)
        % mod EV
        ve_mod(i, j) = res{i, j}.ve_mod;

        % geometry indices
        if res{i, j}.mC_opt>0
            Ahat = mean(res{i, j}.cvA.A, 3);
            WShat = mean(res{i, j}.cv_mod.WS, 3);
            WYhat = mean(res{i, j}.cv_mod.WY, 3);

            alpha_hat = MIC.compute_alphas(Ahat, res{i, j}.mA_opt, WShat, WYhat);
            nshufs = 1000;
            alpha_shuf = nan(nshufs, 4);
            for k = 1:nshufs
                abc = MIC.compute_alphas(Ahat, res{i, j}.mA_opt, randn(size(WShat)), randn(size(WYhat)));
                alpha_shuf(k, :) = [mean(abc.in) mean(abc.Srot) mean(abc.Yrot) mean(abc.new)]; 
            end
    
            alpha_sub(i, j, :) = [mean(alpha_hat.in) mean(alpha_hat.Srot) mean(alpha_hat.Yrot) mean(alpha_hat.new)] - mean(alpha_shuf, 1);
        end
    end
end

load(@@ path to list of sessions with significant modulation); %-->keep

% plot
figure;
subplot(2, 1, 1); hold on;
for i = 1:size(ve_mod, 2)
    xx = (1:28)+0.08*randn(1, 28);
    scatter(xx, ve_mod(:, i), 40, 'k', 'filled', 'o', 'MarkerFaceAlpha', 0.2); 
end

ylabel('mod. EV', 'FontSize', 16)

cols = [0.6 0.7799 0.2420 ; 0.2917 0.0396 0.9867 ; 0.6950 0.4805 0.1425 ; 0.8601 0.1049 0.4989];

subplot(2, 1, 2); hold on;
for i = 1:size(ve_mod, 2)
    for j = 1:4
        xx = (1:28)+0.08*randn(1, 28); xx = xx(keep);
        scatter(xx, alpha_sub(keep, i, j), 40, cols(j, :), 'filled', 'o', 'MarkerFaceAlpha', 0.2); 
    end
end

ylabel('\alpha (null subtr.)', 'FontSize', 16)
xlabel('session #', 'FontSize', 16)


%%%%%%%
% (c-e)
%%%%%%%

% fitting parameters (MICs)
params = struct;
params.max_lin_rank = 20;
params.K = 10;
params.ctgs_folds_on = true;
params.seed = 42; 
params.opts = struct;
params.stratify_by = [];
params.max_mod_rank = 10;

% fitting parameters (quantile split of each component of Z)
Nsplits = 2:5;

EV_all = cell(28, length(Nsplits));

for idx = 1:28
    % load S, X, Y
    disp(idx);
    rng("shuffle");
    [ACA, VIP, wheel, pupil, stim_on, t] = readACAVIP(idx); % read data
    disp(size(VIP)); disp(size(ACA))
    X = [wheel pupil];
    S = ACA;
    Y = VIP;

    kp = sum(isnan([S X Y]), 2)==0;
    X = X(kp, :); S = S(kp, :); Y = Y(kp, :);

    % use different numbers of splits per component
    for aaa = 1:length(Nsplits)
        disp([idx aaa])
        % identifier = index of split occupied by each sample
        identifier = equipopulated_bins_2d(X, Nsplits(aaa), Nsplits(aaa));
    
        EV_A_split = modulator_split_no_stratify(S, X, Y, identifier, params.K, Nsplits(aaa), params.seed);
        EV_all{idx, aaa} = EV_A_split;
    end

end



load(@ path to fig 4 ACAxVIP results); %-->saver
saverMIC = saver;

evA = [];
evMultipleA = [];
evMod = [];
evMod_MIC = [];
% need to subtract EV_{AIS} to get EV_{mod}
for i = 1:size(EV_all, 1)
    evA(i) = saverMIC.res{i}.veA(saverMIC.res{i}.mA_try==saverMIC.res{i}.mA_opt);
    evMod_MIC(i) = saverMIC.res{i}.ve_mod(saverMIC.res{i}.mC_try==saverMIC.res{i}.mC_opt);
    for j = 1:size(EV_all, 2)
        evMultipleA(i, j) = nanmean(EV_all{i, j});
        evMod(i, j) = nanmean(EV_all{i, j})-evA(i);
    end
end


figure;
subplot(1, 2, 1);
hold on;
for i = 1:size(evMod, 2)
    scatter(i+1+randn(size(evMod, 1), 1)*0.08, evMod(:, i), 40, 'k', 'filled', 'MarkerFaceAlpha', 0.3);
end
my_plotter(2:5, evMod, 'k', 'r');
sam_style;
xlim([1.5 5.5]);
xticks(2:5);

subplot(1, 2, 2); 
hold on;
evMod_selected = max([zeros(size(evMod, 1), 1) evMod], [], 2);
scatter(randn(size(evMod, 1), 1)*0.08, evMod_selected, 40, 'k', 'filled', 'MarkerFaceAlpha', 0.3);
my_plotter(0, evMod_selected, 'k', 'r');
scatter(0.75+randn(size(evMod, 1), 1)*0.08, evMod_MIC, 40, 'k', 'filled', 'MarkerFaceAlpha', 0.3);
my_plotter(0.75, evMod_MIC', 'k', 'r');
sam_style;


function my_plotter(x, y, col_line, col_mkr)
mu = mean(y, 1); sem = std(y, [], 1)/sqrt(size(y, 1));
scatter(x, mu, 100, col_mkr, 'filled', 'MarkerEdgeColor', col_line, 'LineWidth', 1.5);
errorbar(x, mu, sem, 'color', col_line, 'LineWidth', 1.5, 'LineStyle', 'none');
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% split pipeline %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% has ctgs_folds_on
function EV_A_split = modulator_split_no_stratify(S, X, Y, identifier, K, Nsplits, seed)

    maxEvals = 10;
    
    % fit B and D without splitting
    mB = 0:min([size(X, 2), size(Y, 2)]); mD = 0:min([size(X, 2), size(S, 2)]);
    cvB = cell(size(mB)); cvD = cell(size(mD));
    veB = nan(size(cvB)); semB = nan(size(cvB)); veD = nan(size(cvD)); semD = nan(size(cvD));
    lambdaB = nan(size(cvB)); lambdaD = nan(size(cvD));
    
    for i = 1:length(mB)
        % important that this is true here - means ctgs_folds_on is indeed
        % on (note it would be more natural to pass this through params argument)
        [cvB{i}, lambdaB(i)] = MIC.select_reg_RRR(K, true, [], seed, maxEvals, X, Y, mB(i));
        veB(i) = cvB{i}.mean_VE;
        semB(i) = std(cvB{i}.ve_fold)/sqrt(K);
    end
    
    for i = 1:length(mD)
        [cvD{i}, lambdaD(i)] = MIC.select_reg_RRR(K, true, [], seed, maxEvals, X, S, mD(i));
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
    cvA_split = cell(1, Nsplits^2);
    for nn = 1:(Nsplits^2)
        disp(nn);
        cvA = cell(size(mA)); veA = nan(size(cvA)); semA = nan(size(cvA)); lambdaA = nan(size(cvA)); 
        for i = 1:length(mA)
            [cvA{i}, lambdaA(i)] = select_reg_A_custom(S, X, Y, mA(i), opt_cvBD, maxEvals, identifier==nn);
            veA(i) = cvA{i}.mean_VE;
            semA(i) = nanstd(cvA{i}.ve_fold)/sqrt(sum(~isnan(cvA{i}.ve_fold))); % important!
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
        for nn = 1:(Nsplits^2)
            Ypred(identifier_test==nn, :) = Xtest(identifier_test==nn, :)*opt_cvBD.B(:, :, k) + (Stest(identifier_test==nn, :)-Xtest(identifier_test==nn, :)*opt_cvBD.D(:, :, k))*cvA_split{nn}.A(:, :, k);
        end

        if sum(isnan(Ypred(:)))==0

            % (B, D) prediction
            Ypred_BD = Xtest*opt_cvBD.B(:, :, k);
            
            % additional explained variance from A
            EV_A_split(k) = MIC.compute_EV(Ytrain, Ytest, Ypred) - MIC.compute_EV(Ytrain, Ytest, Ypred_BD);
        end
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
        
        if (sum(keep_train)>mA) & (sum(keep_test)>0)
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
        else
            Aall(:, :, k) = nan;
            ve_fold(k) = nan;
        end

    end
else
    disp("Warning: zero-rank A-model (cv_fit_A).");
end

cv = struct();
cv.ve_fold = ve_fold;
cv.mean_VE = mean(ve_fold, 'omitnan'); % IMPORTANT IN THIS CASE
cv.cvBD = cvBD;
cv.A = Aall;
cv.lambdaA = lambdaA;
cv.mA = mA;

end


function identifier = equipopulated_bins_2d(X, n1, n2)

N = size(X,1);
identifier = nan(N,1);

% Equal-population bins along first coordinate
[~, order1] = sort(X(:,1));

for i = 1:n1
    idx_start = floor((i-1)*N/n1) + 1;
    idx_end   = floor(i*N/n1);

    idx1 = order1(idx_start:idx_end);

    % Within this first-coordinate bin, sort by second coordinate
    [~, local_order] = sort(X(idx1,2));
    idx1 = idx1(local_order);

    Ni = length(idx1);

    for j = 1:n2
        j_start = floor((j-1)*Ni/n2) + 1;
        j_end   = floor(j*Ni/n2);

        idx2 = idx1(j_start:j_end);
        identifier(idx2) = n2*(i-1) + j;
    end
end
end


%%%%%%%
% (f)
%%%%%%%

