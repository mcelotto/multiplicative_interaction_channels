function [cv_best, lambda] = select_reg_RRR(K, ctgs_folds_on, stratify_by, seed, maxEvals, X, Y, m) %+MIC

if m ==0
    disp("Warning: zero-rank RRR model (select_reg_RRR).");
    lambda = 0;
    cv_best = MIC.cv_fit_RRR(K, ctgs_folds_on, stratify_by, seed, X, Y, m, 0);
    return;
end

% choose range of regularization penalties to try
XX = zscore(X, [], 1);
lambdas = MIC.select_penalty_range(XX);

% Bayesian optimization of regularization penalties
% search space
v_lambda  = optimizableVariable('lambda', lambdas, 'Transform','log');

objFcn = @(T) -MIC.cv_fit_RRR(K, ctgs_folds_on, stratify_by, seed, X, Y, m, T.lambda).mean_VE; % make sure signature matches

opt_results = bayesopt(objFcn, v_lambda, ...
        'IsObjectiveDeterministic', true, ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'MaxObjectiveEvaluations', maxEvals, ...
        'UseParallel', false, 'Verbose', 0, 'PlotFcn', []); % remove 'PlotFcn', [] to show opt plots

lambda = opt_results.XAtMinObjective.lambda;

% refit best model
cv_best = MIC.cv_fit_RRR(K, ctgs_folds_on, stratify_by, seed, X, Y, m, lambda);

% check whether selected lambdas are at boundary of search space
if log(lambda/lambdas(1))<0.1
    disp("Warning: bumping up against lambda lower boundary.")
elseif log((lambdas(end)/lambda))<0.1
    disp("Warning: bumping up against lambda upper boundary.")
end

end