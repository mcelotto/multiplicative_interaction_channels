function lambdas = select_penalty_range(X_standardized)

dMaxShrink = [0.99 0.01];

d = eig(X_standardized'*X_standardized);
dMax = max(d);
lambdas = dMax*(1 - dMaxShrink) ./ dMaxShrink;

end