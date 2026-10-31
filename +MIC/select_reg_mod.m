function [cv_best, lambdaC] = select_reg_mod(cv_best_A, S, X, Y, mC, seed, maxEvals, opts) % MIC

if isinteger(mC) & mC == 0
    disp("Warning: zero-rank modulation model (select_reg_mod).");
    lambdaC = 0;
    cv_best = MIC.cv_fit_mod(cv_best_A, S, X, Y, mC, lambdaC, seed, opts);
    return;
end

% choose range of regularization penalties to try
SS = zscore(S, [], 1); XX = zscore(X, [], 1);
SS = SS - XX*mean(cv_best_A.cvBD.D, 3);
interaction = nan(size(SS, 1), size(SS, 2)*size(XX, 2));
for i = 1:size(interaction, 1)
    interaction(i, :) = vec(SS(i, :)'*XX(i, :))';
end

lambdasC = MIC.select_penalty_range(interaction);

% Bayesian optimization of regularization penalties
% search space
v_lambdaC  = optimizableVariable('lambdaC', lambdasC, 'Transform','log');
% objective function
if strcmp(mC, "full")
    disp("optimizing with full rank...");
    objFcn = @(T) -MIC.cv_fit_mod_full_rank(cv_best_A, S, X, Y, T.lambdaC, seed).mean_VE;
else
    objFcn = @(T) -MIC.cv_fit_mod(cv_best_A, S, X, Y, mC, T.lambdaC, seed, opts).mean_VE;
end

opt_results = bayesopt(objFcn, v_lambdaC, ...
        'IsObjectiveDeterministic', true, ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'MaxObjectiveEvaluations', maxEvals, ...
        'UseParallel', false, 'Verbose', 0, 'PlotFcn', []); % remove 'PlotFcn', [] to show opt plots

lambdaC = opt_results.XAtMinObjective.lambdaC;

% refit best model
if strcmp(mC, "full")
    cv_best = MIC.cv_fit_mod_full_rank(cv_best_A, S, X, Y, lambdaC, seed);
else
    cv_best = MIC.cv_fit_mod(cv_best_A, S, X, Y, mC, lambdaC, seed, opts);
end


% check whether selected lambdas are at boundary of search space
if log(lambdaC/lambdasC(1))<0.1
    disp("Warning: bumping up against lambdaC lower boundary.")
elseif log((lambdasC(end)/lambdaC))<0.1
    disp("Warning: bumping up against lambdaC upper boundary.")
end

end


function avec = vec(a)
avec = a(:);
end