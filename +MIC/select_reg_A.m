function [cv_best, lambdaA] = select_reg_A(S, X, Y, mA, cvBD, maxEvals) %+MIC

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
    cv_best = MIC.cv_fit_A(cvBD, S, X, Y, mA, lambdaA);
    return;
else
    objFcn = @(T) -MIC.cv_fit_A(cvBD, S, X, Y, mA, T.lambdaA).mean_VE;
end


opt_results = bayesopt(objFcn, v_lambdaA, ...
        'IsObjectiveDeterministic', true, ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'MaxObjectiveEvaluations', maxEvals, ...
        'UseParallel', false, 'Verbose', 0, 'PlotFcn', []); % remove 'PlotFcn', [] to show opt plots

lambdaA = opt_results.XAtMinObjective.lambdaA;

% refit best model
cv_best = MIC.cv_fit_A(cvBD, S, X, Y, mA, lambdaA);

% check whether selected lambdas are at boundary of search space
if log(lambdaA/lambdasA(1))<0.1
    disp("Warning: bumping up against lambdaA lower boundary.")
elseif log((lambdasA(end)/lambdaA))<0.1
    disp("Warning: bumping up against lambdaA upper boundary.")
end

end