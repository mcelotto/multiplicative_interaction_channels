function [WS, WX, WY] = fit_mod(S, X, Y, mC, lambdaC, seed) % +MIC
% set seed for reproducability
rng(seed);

%repeat until convergence criterion is met
nS = size(S, 2); nX = size(X, 2); nY = size(Y, 2);

% check edge cases
if mC==0
    disp("Warning: zero-rank modulation (fit_mod).")
    WS = nan; WY = nan; WX = nan;
    return;
end

rank_upper_bound = max([nS*nY nS*nX nX*nY]);
if mC>rank_upper_bound
    mC = rank_upper_bound;
    disp("Warning: modulation rank exceeds max (fit_mod).")
end

% random initialization
WS = randn(nS, mC); WX = randn(nX, mC); WY = randn(nY, mC);

%convergence criteria
maxIter = 100; eps = 1e-5; patience = 3; bad = 0;
Lprev = obj_value(S, X, Y, WS, WX, WY, lambdaC);
for iter = 1:maxIter
    [WS, WX, WY] = mod_als_sweep_speedup(S, X, Y, WS, WX, lambdaC); % change back

    %check for convergence
    L = obj_value(S, X, Y, WS, WX, WY, lambdaC);

    rel = (Lprev - L) / max(1, Lprev);
    if rel < eps
        bad = bad + 1;
        if bad >= patience, break; end
    else
        bad = 0;
    end
    Lprev = L;
end
end

function L = obj_value(S, X, Y, WS, WX, WY, lambdaC)
    Ypred = ((S*WS).*(X*WX))*WY.';
    R = Y - Ypred;

    % calculate Frobenius norm of C with Gram matrices
    GY = WY.' * WY;   % mC x mC, Gram matrix for WY columns
    GS = WS.' * WS;   % mC x mC, Gram matrix for WS columns
    GX = WX.' * WX;   % mC x mC, Gram matrix for WX columns
    normC = sqrt(sum(GY(:) .* GS(:) .* GX(:)));

    L = sum(R(:).^2) + lambdaC*normC^2;
end


function [WS, WX, WY] = mod_als_sweep_speedup(S, X, Y, WS, WX, lambdaC)
%MOD_ALS_SWEEP One ALS sweep for modulation model with BIG speedups:
%  (1) Remove diag(...) via weighted-matrix identity
%  (2) Build only half of symmetric block systems + mirror
%  (3) Use Cholesky-based solves via decomposition('chol') when possible

% ---------- dims ----------
[N, pS] = size(S);
[~, pX] = size(X);
R = size(WS, 2);

% ---------- common projections ----------
KX = X * WX;          % N x R
KS = S * WS;          % N x R
Phi = KS .* KX;       % N x R

GS = WS' * WS;        % R x R
GX = WX' * WX;        % R x R

% ======================================================================
% update WY  (solve: (Phi'Phi + lambdaC*(GS.*GX)) * WY' = Phi'Y)
% ======================================================================
Awy = Phi' * Phi + lambdaC * (GS .* GX);     % R x R
Bwy = Phi' * Y;                               % R x pY

% Prefer Cholesky if SPD; otherwise fall back to backslash
try
    dAwy = decomposition(Awy, 'chol');
    WYt  = dAwy \ Bwy;                        % R x pY  (this is WY')
catch
    WYt  = Awy \ Bwy;
end
WY = WYt.';                                   % pY x R

% ======================================================================
% update WS  (solve: Gs * vec(WS) = vec(S'*(KY.*KX)))
% big speedups: no diag(), symmetry, chol decomposition
% ======================================================================
GY = WY' * WY;        % R x R (symmetric)
KY = Y * WY;          % N x R
L  = GY .* GX;        % R x R

H = S' * (KY .* KX);  % pS x R
b = H(:);             % vec by columns

% Precompute weighted S: Sx(:,:,j) = S .* KX(:,j)
Sx = zeros(N, pS, R);
for j = 1:R
    Sx(:,:,j) = S .* KX(:,j);
end

Gs = zeros(pS*R, pS*R);
I  = eye(pS);

for j = 1:R
    rows = (pS*(j-1)+1):(pS*j);
    for r = 1:j
        cols = (pS*(r-1)+1):(pS*r);

        % S' * diag(KX(:,r).*KX(:,j)) * S  ==  (S.*KX(:,r))' * (S.*KX(:,j))
        A = Sx(:,:,r)' * Sx(:,:,j);                          % pS x pS
        Bij = GY(r,j) * A + lambdaC * L(r,j) * I;            % pS x pS

        Gs(rows, cols) = Bij;
        if r ~= j
            Gs(cols, rows) = Bij.';                          % mirror
        end
    end
end

try
    dGs = decomposition(Gs, 'chol');
    w   = dGs \ b;
catch
    w   = Gs \ b;
end
WS = reshape(w, pS, R);

% ======================================================================
% update WX  (solve: Gx * vec(WX) = vec(X'*(KY.*KS)))
% same speedups: no diag(), symmetry, chol decomposition
% ======================================================================
KS = S * WS;          % update KS since WS changed
GS = WS' * WS;        % update GS

L  = GY .* GS;        % R x R

H = X' * (KY .* KS);  % pX x R
b = H(:);

% Precompute weighted X: Xs(:,:,j) = X .* KS(:,j)
Xs = zeros(N, pX, R);
for j = 1:R
    Xs(:,:,j) = X .* KS(:,j);
end

Gx = zeros(pX*R, pX*R);
I  = eye(pX);

for j = 1:R
    rows = (pX*(j-1)+1):(pX*j);
    for r = 1:j
        cols = (pX*(r-1)+1):(pX*r);

        % X' * diag(KS(:,r).*KS(:,j)) * X  ==  (X.*KS(:,r))' * (X.*KS(:,j))
        A = Xs(:,:,r)' * Xs(:,:,j);                          % pX x pX
        Bij = GY(r,j) * A + lambdaC * L(r,j) * I;            % pX x pX

        Gx(rows, cols) = Bij;
        if r ~= j
            Gx(cols, rows) = Bij.';                          % mirror
        end
    end
end

try
    dGx = decomposition(Gx, 'chol');
    w   = dGx \ b;
catch
    w   = Gx \ b;
end
WX = reshape(w, pX, R);
end
