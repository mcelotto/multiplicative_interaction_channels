function out = bilinearity_test(sX, sZ, sY, foldid) % +MIC

% Held-out diagnostic of the bilinearity of a fitted MIC (Appendix B.11).
% The target score sY is regressed on the bilinear term alone (null) and
% on the bilinear term plus higher-order terms in one factor:
%   bilinear (null):  sY ~ sX.*sZ
%   nonlinear in x:   sY ~ sX.*sZ + sX.^2.*sZ + sX.^3.*sZ
%   nonlinear in z:   sY ~ sX.*sZ + sX.*sZ.^2 + sX.*sZ.^3
% All regressions are ridge-regularized and evaluated on held-out folds.
% The extra penalty on the higher-order terms is selected by inner
% cross-validation within each training fold.
%
% NOTE ON VARIABLE NAMING: sX, sZ, sY follow the paper (source, modulator,
% target scores).
%
% Inputs:
%   sX, sZ, sY  [N x 1]  held-out channel scores (see heldout_scores.m)
%   foldid      [N x 1]  CV fold of each sample
%
% Outputs:
%   out.X.dve   held-out Delta EV_NL for nonlinearity in x:
%               (RSS_null - RSS_alt) / TSS; > 0 indicates a violation of
%               bilinearity in x
%   out.Z.dve   same for nonlinearity in z
%   Both are NaN if the bilinear term explains less than modVE_min of the
%   held-out variance of sY (no modulation to test).
% Significance is assessed outside this function (sign test across
% simulation runs or sessions).

%%%%%%%%%%%%%%%%%%%%%%%
modVE_min = 0.001; % minimum held-out EV of the bilinear term
winsSD = 6; % scores winsorized at mean +- winsSD*SD
ridge = 1e-3; % base ridge on all coefficients (scaled by trace(F'*F)/p)
lambda_grid = [0 10.^(-4:1)]; % candidate extra penalties on the higher-order terms
K_inner = 5; % inner CV folds for penalty selection
%%%%%%%%%%%%%%%%%%%%%%%

% remove nans, winsorize
valid = ~isnan(sX) & ~isnan(sZ) & ~isnan(sY);
sX = winsor(sX(valid), winsSD);
sZ = winsor(sZ(valid), winsSD);
sY = sY(valid);
foldid = foldid(valid);

% features
f1 = sX .* sZ; % bilinear
BX = sX.^(2:3) .* sZ; % nonlinear in x
BZ = sX .* sZ.^(2:3); % nonlinear in z

% total held-out sum of squares
u = unique(foldid);
TSS = 0;
for k = 1:numel(u)
    te = foldid == u(k);
    TSS = TSS + sum((sY(te) - mean(sY(~te))).^2);
end

% null model and gate
RSS0 = heldout_rss(f1, sY, foldid, ridge, lambda_grid, K_inner);
SS_lin = TSS - RSS0;
gate = SS_lin <= 0 || SS_lin/TSS < modVE_min;

out.X.dve = NaN;
out.Z.dve = NaN;
if ~gate
    out.X.dve = (RSS0 - heldout_rss([f1, BX], sY, foldid, ridge, lambda_grid, K_inner)) / TSS;
    out.Z.dve = (RSS0 - heldout_rss([f1, BZ], sY, foldid, ridge, lambda_grid, K_inner)) / TSS;
end
end

%%%%%%% HELPERS %%%%%%
function rss = heldout_rss(F, y, fid, ridge, lambda_grid, K_inner)
% held-out RSS of ridge regression of y on [1 F], features standardized
% on each training fold. Columns 2:end of F (higher-order terms) get an
% extra penalty lambda*n_train, with lambda selected by inner CV.
u = unique(fid);
rss = 0;
tune = size(F, 2) > 1;
for k = 1:numel(u)
    te = fid == u(k);
    tr = ~te;
    mu = mean(F(tr, :), 1);
    sd = std(F(tr, :), 0, 1);
    sd(sd < 1e-12) = 1;
    Ftr = [ones(sum(tr), 1), (F(tr, :) - mu) ./ sd];
    Fte = [ones(sum(te), 1), (F(te, :) - mu) ./ sd];
    ytr = y(tr);
    lam = 0;
    if tune
        lam = select_lambda(Ftr, ytr, ridge, lambda_grid, K_inner);
    end
    b = ridge_solve(Ftr, ytr, lam, ridge);
    rss = rss + sum((y(te) - Fte * b).^2);
end
end

function lam = select_lambda(F1, y, ridge, lambda_grid, K_inner)
% extra penalty with the smallest held-out RSS across K_inner temporally
% contiguous folds of the training data
n = size(F1, 1);
edges = round(linspace(0, n, K_inner + 1));
err = zeros(1, numel(lambda_grid));
for m = 1:K_inner
    te = false(n, 1);
    te(edges(m)+1:edges(m+1)) = true;
    tr = ~te;
    for g = 1:numel(lambda_grid)
        b = ridge_solve(F1(tr, :), y(tr), lambda_grid(g), ridge);
        err(g) = err(g) + sum((y(te) - F1(te, :) * b).^2);
    end
end
[~, ig] = min(err);
lam = lambda_grid(ig);
end

function b = ridge_solve(F1, y, lam, ridge)
% F1 = [1, standardized features]: base ridge on all coefficients, plus
% lam*n on the higher-order terms (columns 3:end, after intercept and
% bilinear term)
XtX = F1' * F1;
p = size(XtX, 1);
pen = ridge * trace(XtX) / p * ones(p, 1);
pen(3:end) = pen(3:end) + lam * size(F1, 1);
b = (XtX + diag(pen)) \ (F1' * y);
end

function v = winsor(v, k)
m = mean(v);
s = std(v);
v = min(max(v, m - k * s), m + k * s);
end
