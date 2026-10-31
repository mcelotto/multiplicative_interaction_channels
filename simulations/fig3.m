%%%%%%%%%%%%%%%%%%%%%%%
% simulations shown in Fig. 3
% validating the geometric decomposition
%%%%%%%%%%%%%%%%%%%%%%%

% simulation settings
nS = 50; nY = 5; nX = 1; % population dimensionalities
mA = 2; % AIS rank
mC = 1; % MIC rank
N = 1000; % number of samples per sim
noise = 1; % emission noise
mod_stren = 1; % MIC strength
num_sims = 30; % number of sims per condition
thetaS_gt = 0; % angle between MIC and AIS in source population 
% NOTE: we ran this script twice: once with thetaS_gt = 0 (Fig. 3b-d)
% and once with thetaS_gt = acos(sqrt(3/4)) (Fig. 3e-g)
thetaY_gt = 0; % angle between MIC and AIS in target population

% fitting parameters
params = struct;
params.max_lin_rank = 10;
params.K = 10;
params.ctgs_folds_on = false; 
params.seed = 42; 
params.opts = struct;
params.max_mod_rank = 5;
params.stratify_by = [];

% store results
alpha_shuf_all = cell(1, num_sims);
alpha_hat_all = cell(1, num_sims);
EVcstr_all = cell(1, num_sims);

for iii = 1:num_sims
    disp(iii); rng("shuffle");

    % randomly oriented AIS
    UA = randn(nS, mA); UA = UA./vecnorm(UA, 2, 1);
    VA = randn(nY, mA); VA = VA./vecnorm(VA, 2, 1);
    A = UA*VA';

    % MIC oriented at specified angles to AIS
    WS = randvec_angle_to_subspace(UA, thetaS_gt);
    WY = randvec_angle_to_subspace(VA, thetaY_gt);      
    WS = mod_stren*WS./vecnorm(WS, 2, 1);
    WY = WY./vecnorm(WY, 2, 1);
    WX = randn(nX, mC); WX = WX./vecnorm(WX, 2, 1);

    % simulate
    S = randn(N, nS);
    X = randn(N, nX);
    Y = S*A + ((S*WS).*(X*WX))*WY.' + noise*randn(N, nY);

    results = MIC.model_wrapper(S, X, Y, params);
    Ahat = mean(results.cvA{results.mA_try==results.mA_opt}.A, 3);
    WShat = mean(results.cv_mod{results.mC_try==results.mC_opt}.WS, 3);
    WYhat = mean(results.cv_mod{results.mC_try==results.mC_opt}.WY, 3);

    % compute geometry indices
    % raw
    abc = MIC.compute_alphas(Ahat, results.mA_opt, WShat, WYhat);
    alpha_hat_all{iii} = [abc.in abc.Srot abc.Yrot abc.new];

    % rotation null
    nshufs = 1000; % number of shuffles used to estimate null distribution
    alpha_shuf = nan(nshufs, size(WShat, 2), 4);
    for i = 1:nshufs
        abc = MIC.compute_alphas(Ahat, results.mA_opt, randn(size(WShat)), randn(size(WYhat)));
        alpha_shuf(i, 1) = abc.in;
        alpha_shuf(i, 2) = abc.Srot;
        alpha_shuf(i, 3) = abc.Yrot;
        alpha_shuf(i, 4) = abc.new;
    end
    alpha_shuf_all{iii} = squeeze(mean(alpha_shuf, 1));

    % constrained MIC fits
    EVcstr = nan(1, 4);
    params2 = params;
    params2.max_mod_rank = results.mC_opt; % in constrained fit, re-select MIC rank, but restrict search range to [0, prev. selected rank]
    params2.mA = results.mA_opt; params2.mB = results.mB_opt; params2.mD = results.mD_opt;

    params2.opts.source_constraint = "same"; params2.opts.target_constraint = "same";
    abc = MIC.model_wrapper(S, X, Y, params2); % confine source and target MIC to AIS
    EVcstr(1) = abc.ve_mod(abc.mC_try==abc.mC_opt);

    params2.opts.source_constraint = "none"; params2.opts.target_constraint = "same";
    abc = MIC.model_wrapper(S, X, Y, params2); % confine target MIC to AIS
    EVcstr(2) = abc.ve_mod(abc.mC_try==abc.mC_opt);

    params2.opts.source_constraint = "same"; params2.opts.target_constraint = "none";
    abc = MIC.model_wrapper(S, X, Y, params2); % confine source MIC to AIS
    EVcstr(3) = abc.ve_mod(abc.mC_try==abc.mC_opt);

    EVcstr(4) = results.ve_mod(results.mC_try==results.mC_opt); % unconstrained

    EVcstr_all{iii} = EVcstr;
end

% plot results
cols = [0.6 0.7799 0.2420 ; 0.2917 0.0396 0.9867 ; 0.6950 0.4805 0.1425 ; 0.8601 0.1049 0.4989];

figure;

% raw geometry indices
subplot(1, 3, 1); hold on;
mu = [];
for i = 1:num_sims
    scatter((1:4)+0.08*randn(1, 4), alpha_hat_all{i}, 30, cols, 'filled', 'o', 'MarkerFaceAlpha', 0.15);
    mu(i, :) = alpha_hat_all{i};
end
scatter(1:4, mean(mu, 1), 100, cols, 'filled', 'o');
scatter(1:4, [1 0 0 0], 100, 'k', 's', 'LineWidth', 1);
xlim([0.5 4.5])
ylim([0 1])

% null-subtracted geometry indices
subplot(1, 3, 2); hold on;
mu = [];
for i = 1:num_sims
    scatter((1:4)+0.08*randn(1, 4), alpha_hat_all{i}(:)'-alpha_shuf_all{i}(:)', 30, cols, 'filled', 'o', 'MarkerFaceAlpha', 0.15);
    mu(i, :) = alpha_hat_all{i}(:)'-alpha_shuf_all{i}(:)';
end
scatter(1:4, mean(mu, 1), 80, cols, 'filled', 'o');
xlim([0.5 4.5])

% constrained fit EV
subplot(1, 3, 3); hold on;
mu = [];
for i = 1:num_sims
    scatter((1:4)+0.08*randn(1, 4), EVcstr_all{i}, 30, [cols ; zeros(3, 3)], 'filled', 'o', 'MarkerFaceAlpha', 0.15);
    mu(i, :) = EVcstr_all{i};
end
scatter(1:4, mean(mu, 1), 80, [cols ; zeros(3, 3)], 'filled', 'o')
xlim([0.5 4.5])

% helpers

% draw a vector uniformly from the unit sphere in the subspace orthogonal
% to the column space of UA
function x = randvec_angle_to_subspace(UA, theta)
    % UA: n x k matrix whose columns span the subspace
    % theta: desired angle in radians, between 0 and pi/2

    [n,~] = size(UA);

    % Orthonormalize basis for the subspace
    Q = orth(UA);

    % Random unit vector in the subspace
    a = randn(size(Q,2),1);
    u = Q*a;
    u = u / norm(u);

    % Random unit vector in the orthogonal complement
    b = randn(n,1);
    v = b - Q*(Q'*b);   % remove subspace component
    v = v / norm(v);

    % Combine to get desired angle
    x = cos(theta)*u + sin(theta)*v;
end