function A = fit_RRR(X, Y, m, lambda)
if m>min(size(X, 2), size(Y, 2))
    disp("Warning: rank restriction exceeds dimensionality.")
    m = min(size(X, 2), size(Y, 2));
end

A_ridge = MIC.ridge_fit(X, Y, lambda);
[~, ~, V] = svds(X*A_ridge, m);
A = A_ridge*V(:, 1:m)*V(:, 1:m).';
end

