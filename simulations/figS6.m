clear; close all;

all_scenarios = {'bilinear', 'rotation'}; % scenarios to test MLPs on
a_idx = {6, [5, 11]}; % values of modulation strength parameter 'a' to use
n_samples_list = [100, 200, 400, 800, 1600, 3200]; % sample counts to use
hlist = [1 10 100]; % hidden layer size h*mC
res = cell(length(all_scenarios), length(hlist)); % store MLP results here
resMIC = cell(length(all_scenarios), 1); % store MICs results here

counter = 1;
for scenario = all_scenarios
    scenario = scenario{1};
    data_dir = @ path to nonlinear simulation results
    rot_setting = 2; % 1 = perp, 2 = par

    switch scenario
        case 'bilinear'
            Ld=load([data_dir 'bilinear_saver.mat']); G=Ld.saver.results; xv=Ld.saver.a_list(:); xlab='a (modulation strength)';
        case 'sigmoid'
            Ld=load([data_dir 'sigmoid_nonlinearity_saver.mat']); G=Ld.saver.results; xv=Ld.saver.a_list(:); xlab='a (modulation strength)';
        case 'relu'
            Ld=load([data_dir 'relu_nonlinearity_saver.mat']);    G=Ld.saver.results; xv=Ld.saver.a_list(:); xlab='a (modulation strength)';
        case 'exp'
            Ld=load([data_dir 'exponential_in_Z_modulation_saver.mat']); G=Ld.saver.results; xv=Ld.saver.a_list(:); xlab='b (modulation strength)';
        case 'rotation'
            Ld=load([data_dir 'rotation_sim_results_2.mat']); G=squeeze(Ld.results_all(rot_setting,:,:)); xv=linspace(0,pi,size(G,1))'; xlab='rotation amplitude';
        case 'relu_on_Z'
            Ld=load([data_dir 'relu_on_Z_directly_saver_take2.mat']); G=Ld.saver.results; xv=Ld.saver.a_list(:); xlab='a (modulation strength)';
        case 'indirect_relu_on_Z'
            Ld=load([data_dir 'indirect_relu_on_Z_saver.mat']); G=Ld.saver.results; xv=Ld.saver.a_list(:); xlab='a (modulation strength)';
        otherwise, error('unknown scenario');
    end

    for hidx = 1:length(hlist)
        res{counter, hidx} = cell([size(G) length(n_samples_list)]);
    end
    resMIC{counter} = cell([size(G) length(n_samples_list)]);

    for i = a_idx{counter}
        for j = 1:size(G, 2)
            X = G{i, j}.S; Y = G{i, j}.Y; Z = G{i, j}.X;

            for nn = 1:length(n_samples_list)
                disp([i, j, nn])
                % fitting setttings
                params = struct;
                params.max_lin_rank = 10;
                params.K = 10;
                params.ctgs_folds_on = true; 
                params.seed = 42; 
                params.opts = struct;
                params.stratify_by = [];
                params.max_mod_rank = 5;

                Xc = X(1:n_samples_list(nn), :);
                Yc = Y(1:n_samples_list(nn), :);
                Zc = Z(1:n_samples_list(nn), :);

                resMIC{counter}{i, j, nn} = MIC.model_wrapper(Xc, Zc, Yc, params);
                mC_opt = res_MIC{i, j, nn}.mC_opt;

                tmp = res_MIC{i, j, nn};
                Ahat = tmp.cvA{tmp.mA_try==tmp.mA_opt}.A;
                Bhat = tmp.cvA{tmp.mA_try==tmp.mA_opt}.cvBD.B;
                Dhat = tmp.cvA{tmp.mA_try==tmp.mA_opt}.cvBD.D;
                cv_partition = tmp.cvA{tmp.mA_try==tmp.mA_opt}.cvBD.cv_partition;

                lambda_list = 10.^((-5):2:3);
                basewidth = 1;
                for hidx = 1:length(hlist)
                    res{counter, hidx}{i, j, nn} = fitMLP(Xc, Zc, Yc, Bhat, Dhat, Ahat, cv_partition, hidden_size=hlist(hidx)*basewidth, lambda_list=lambda_list, mini_batch_size=32);
                end
            end
        end
    end
    counter = counter + 1;
end


% store EV values in abc, abcMIC
abc = cell(size(res)); abcMIC = cell(size(resMIC));
for c = 1:size(abc, 1)
    abcMIC{c} = nan(size(resMIC{c}));
    for hidx = 1:size(abc, 2)
        abc{c, hidx} = nan(size(res{c, hidx}));
        for i = 1:size(res{c, hidx}, 1) %G, 1
            for j = 1:size(res{c, hidx}, 2) %G, 2
                for nn = 1:length(n_samples_list)
                    if ~isempty(res{c, hidx}{i, j, nn})
                        abc{c, hidx}(i, j, nn) = max(res{c, hidx}{i, j, nn}.mean_delta_EV, 0);
                        if hidx==1
                            abcMIC{c}(i, j, nn) = resMIC{c}{i, j, nn}.ve_mod(resMIC{c}{i, j, nn}.mC_try==resMIC{c}{i, j, nn}.mC_opt);
                        end
                    end
                end
            end
        end
    end
end


% plot results
figure; hold on;
cols = winter(length(hlist));
for hidx = 1:length(hlist)
    abc1 = abc{1, hidx}(a_idx{1}, :, :); % repetition x sample count
    abc2 = abc{2, hidx}(a_idx{2}(1), :, :);
    abc3 = abc{2, hidx}(a_idx{2}(2), :, :);

    subplot(1, 3, 1);
    errorbar(n_samples_list, mean(abc1, 1), std(abc1, [], 1)/sqrt(size(abc1, 1)), 'LineWidth', 1.5, 'Color', cols(hidx, :));

    subplot(1, 3, 2);
    errorbar(n_samples_list, mean(abc2, 1), std(abc2, [], 1)/sqrt(size(abc2, 1)), 'LineWidth', 1.5, 'Color', cols(hidx, :));

    subplot(1, 3, 3);
    errorbar(n_samples_list, mean(abc3, 1), std(abc3, [], 1)/sqrt(size(abc3, 1)), 'LineWidth', 1.5, 'Color', cols(hidx, :));

end

for i = 1:3
    subplot(1, 3, i);
    set(gca, 'xscale', 'log');
    xlim([min(n_samples_list)/1.5 max(n_samples_list)*1.5]);
    xlabel("# samples, N");
    ylabel("modulation EV")
    sam_style;
end
legend('h = 1', 'h = 10', 'h = 100', 'h = 1000', 'MICs', 'FontSize', 14);


% MLP fitting function
function res = fitMLP(X, Z, Y, B, D, A, cv_partition, opts)
%FITMLP Fit an MLP to residualized variables using predefined outer folds.
%
% For fold k:
%   X2 = X - Z*D(:,:,k)
%   Y2 = Y - X2*A(:,:,k) - Z*B(:,:,k)
%
% One global lambda is selected by maximizing the mean across outer folds of
%
%   EV(full model on Y) - EV(linear model on Y).

arguments
    X double
    Z double
    Y double
    B double
    D double
    A double
    cv_partition

    opts.use_inner_validation logical = true
    opts.inner_split_type string = "contiguous"
    opts.inner_val_fraction double = 0.2
    opts.seed double = 42

    opts.lambda_list double = 10.^(-5:-1)
    opts.hidden_size double = 20
    opts.max_epochs double = 100
    opts.validation_patience double = 20
    opts.learning_rate double = 1e-3
    opts.mini_batch_size double = 128
    opts.execution_environment string = "cpu"
    opts.show_progress logical = true
end

N = size(X,1);
nX = size(X,2);
nZ = size(Z,2);
nY = size(Y,2);

K = cv_partition.NumTestSets;
nLambda = numel(opts.lambda_list);

assert(cv_partition.NumObservations == N);
assert(size(Z,1) == N && size(Y,1) == N);
assert(isequal(size(D), [nZ nX K]));
assert(isequal(size(A), [nX nY K]));
assert(isequal(size(B), [nZ nY K]));

Y2hat_all = cell(K,nLambda);
Yhat_linear_fold = cell(K,1);
Yhat_full_all = cell(K,nLambda);

EV_linear = nan(K,1);
EV_full = nan(K,nLambda);
delta_EV = nan(K,nLambda);

test_index_all = cell(K,1);

layers = [
    featureInputLayer(nX+nZ, Normalization="none")
    fullyConnectedLayer(opts.hidden_size)
    reluLayer
    fullyConnectedLayer(nY)
];

parfor k = 1:K

    if opts.show_progress
        fprintf("Fold %d/%d\n", k, K);
    end

    trainIdx = find(training(cv_partition,k));
    testIdx = find(test(cv_partition,k));

    test_index_all{k} = testIdx;

    % Normalize before applying A, B, and D.
    [Xtr, Xte] = split_normalize(X, trainIdx, testIdx);
    [Ztr, Zte] = split_normalize(Z, trainIdx, testIdx);
    [Ytr, Yte] = split_normalize(Y, trainIdx, testIdx);

    Ak = A(:,:,k);
    Bk = B(:,:,k);
    Dk = D(:,:,k);

    X2tr = Xtr - Ztr*Dk;
    X2te = Xte - Zte*Dk;

    Y2tr = Ytr - X2tr*Ak - Ztr*Bk;
    Y2te = Yte - X2te*Ak - Zte*Bk;

    if opts.use_inner_validation

        [fitIdx, valIdx] = inner_split( ...
            trainIdx, ...
            opts.inner_val_fraction, ...
            opts.inner_split_type, ...
            opts.seed + k - 1);

        Xfit = X2tr(fitIdx,:);
        Zfit = Ztr(fitIdx,:);
        Yfit = Y2tr(fitIdx,:);

        Xval = X2tr(valIdx,:);
        Zval = Ztr(valIdx,:);
        Yval = Y2tr(valIdx,:);

    else

        Xfit = X2tr;
        Zfit = Ztr;
        Yfit = Y2tr;

        Xval = X2te;
        Zval = Zte;
        Yval = Y2te;
    end

    % Linear prediction is independent of lambda.
    YhatLinear = X2te*Ak + Zte*Bk;

    Yhat_linear_fold{k} = YhatLinear;
    EV_linear(k) = compute_EV(Ytr, Yte, YhatLinear);

    for j = 1:nLambda

        lambda = opts.lambda_list(j);

        if opts.show_progress
            fprintf("    lambda = %.3g\n", lambda);
        end

        % Same initialization across lambdas within each fold.
        rng(opts.seed + 1000*k);

        trainOpts = trainingOptions("adam", ...
            InitialLearnRate=opts.learning_rate, ...
            MaxEpochs=opts.max_epochs, ...
            MiniBatchSize=opts.mini_batch_size, ...
            L2Regularization=lambda, ...
            ValidationData={[Xval Zval], Yval}, ...
            ValidationPatience=opts.validation_patience, ...
            OutputNetwork="best-validation", ...
            Shuffle="every-epoch", ...
            ExecutionEnvironment=opts.execution_environment, ...
            Verbose=false, ...
            Plots="none");

        net = trainnet( ...
            [Xfit Zfit], Yfit, layers, "mse", trainOpts);

        Y2hat = minibatchpredict( ...
            net, [X2te Zte], ...
            MiniBatchSize=opts.mini_batch_size);

        YhatFull = YhatLinear + Y2hat;

        Y2hat_all{k,j} = Y2hat;
        Yhat_full_all{k,j} = YhatFull;

        EV_full(k,j) = compute_EV(Ytr, Yte, YhatFull);
        delta_EV(k,j) = EV_full(k,j) - EV_linear(k);
    end
end

% Select one lambda for the entire dataset.
mean_delta_EV_by_lambda = mean(delta_EV, 1, "omitnan");
[mean_delta_EV, bestLambdaIdx] = max(mean_delta_EV_by_lambda);
best_lambda = opts.lambda_list(bestLambdaIdx);

% Assemble out-of-fold predictions for the selected lambda.
Y2hat = nan(size(Y));
Yhat_linear = nan(size(Y));
Yhat_full = nan(size(Y));

for k = 1:K
    testIdx = test_index_all{k};

    Y2hat(testIdx,:) = Y2hat_all{k,bestLambdaIdx};
    Yhat_linear(testIdx,:) = Yhat_linear_fold{k};
    Yhat_full(testIdx,:) = Yhat_full_all{k,bestLambdaIdx};
end

res.Y2hat = Y2hat;
res.Yhat_linear = Yhat_linear;
res.Yhat_full = Yhat_full;

res.best_lambda = best_lambda;
res.best_lambda_index = bestLambdaIdx;

res.EV_linear_fold = EV_linear;
res.EV_full_fold = EV_full(:,bestLambdaIdx);
res.delta_EV_fold = delta_EV(:,bestLambdaIdx);

res.mean_EV_linear = mean(EV_linear, "omitnan");
res.mean_EV_full = mean(EV_full(:,bestLambdaIdx), "omitnan");
res.mean_delta_EV = mean_delta_EV;

res.mean_delta_EV_by_lambda = mean_delta_EV_by_lambda;
res.EV_full_by_lambda = EV_full;
res.delta_EV_by_lambda = delta_EV;

res.cv_partition = cv_partition;
res.options = opts;

end


function EV = compute_EV(Ytrain, Ytest, Ypred)

R_model = Ytest - Ypred;
R_base = Ytest - mean(Ytrain,1);

SSE = sum(R_model(:).^2);
SST = sum(R_base(:).^2);

EV = 1 - SSE/SST;

end


function [Xtr, Xte] = split_normalize(X, tr, te)

Xtr = X(tr,:);
Xte = X(te,:);

mu = mean(Xtr,1);
sigma = std(Xtr,[],1);
sigma(sigma == 0) = 1;

Xtr = (Xtr - mu)./sigma;
Xte = (Xte - mu)./sigma;

end


function [trainIdx, valIdx] = inner_split( ...
    outerTrainIdx, valFraction, splitType, seed)

n = numel(outerTrainIdx);
nVal = max(1, round(valFraction*n));

if nVal >= n
    error("Validation set must be smaller than training set.");
end

valIdx = false(n,1);

switch lower(splitType)

    case "random"
        oldRng = rng;
        cleanup = onCleanup(@() rng(oldRng)); %#ok<NASGU>
        rng(seed);
        positions = randperm(n,nVal);

    case "contiguous"
        [~, order] = sort(outerTrainIdx);

        maxStart = n - nVal + 1;
        first = 1 + mod(seed - 1, maxStart);

        positions = order(first:first+nVal-1);

    otherwise
        error('inner_split_type must be "random" or "contiguous".');
end

valIdx(positions) = true;
trainIdx = ~valIdx;

end
