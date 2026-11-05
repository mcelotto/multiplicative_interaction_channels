%%%%%%%%%%%%%%%%%%%%%%%
% simulations shown in Fig. S7i-k
% validating geometry decomposition with multiple MICs
%%%%%%%%%%%%%%%%%%%%%%%

% simulation details
nS = 20; nY = 20; nX = 10; % pop dimensionalities
N = 1000; % number of samples per sim
noise = 1; % emission noise
mod_stren = 1; % MIC strength
mA = 2; mClist = 1:3; % ground-truth AIS and MIC ranks
num_sims = 30; % number of simulations per condition

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
alpha_hat_all = cell(length(mClist), num_sims);
alpha_shuf_all = cell(length(mClist), num_sims);
EVcstr_all = nan(length(mClist), num_sims, 4);
thetaS_gt = cell(length(mClist), num_sims);
thetaY_gt = cell(length(mClist), num_sims);

% iterate over different ground-truth MIC ranks
for kkk = 1:length(mClist)
    mC = mClist(kkk);
    for iii = 1:num_sims
        disp([kkk, iii]); rng("shuffle");

        % randomly oriented AIS
        UA = randn(nS, mA); UA = UA./vecnorm(UA, 2, 1);
        VA = randn(nY, mA); VA = VA./vecnorm(VA, 2, 1);
        A = UA*VA';

        % MIC oriented at uniformly distributed angles relative to AIS
        thetaS_gt{kkk, iii} = rand(1, mC)*pi/2; thetaY_gt{kkk, iii} = rand(1, mC)*pi/2;
        WS = []; WY = [];
        for abra = 1:mC
            WS(:, abra) = randvec_angle_to_subspace(UA, thetaS_gt{kkk, iii}(abra));
            WY(:, abra) = randvec_angle_to_subspace(VA, thetaY_gt{kkk, iii}(abra));      
        end
        WS = mod_stren*WS./vecnorm(WS, 2, 1);
        WY = WY./vecnorm(WY, 2, 1);
        WX = randn(nX, mC); WX = WX./vecnorm(WX, 2, 1);

        % simulate
        S = randn(N, nS);
        X = randn(N, nX);
        Y = S*A + ((S*WS).*(X*WX))*WY.' + noise*randn(N, nY);

        % fit
        results = MIC.model_wrapper(S, X, Y, params);
        Ahat = mean(results.cvA{results.mA_try==results.mA_opt}.A, 3);
        WShat = mean(results.cv_mod{results.mC_try==results.mC_opt}.WS, 3);
        WYhat = mean(results.cv_mod{results.mC_try==results.mC_opt}.WY, 3);

        % geometry indices
        nshufs = 1000;
        alpha_shuf = nan(nshufs, size(WShat, 2), 4);
        for i = 1:nshufs
            abc = MIC.compute_alphas(Ahat, results.mA_opt, randn(size(WShat)), randn(size(WYhat)));
            alpha_shuf(i, :, 1) = abc.in;
            alpha_shuf(i, :, 2) = abc.Srot;
            alpha_shuf(i, :, 3) = abc.Yrot;
            alpha_shuf(i, :, 4) = abc.new;
        end

        alpha_shuf = squeeze(mean(alpha_shuf, 1));
        alpha_shuf_all{kkk, iii} = alpha_shuf;

        abc = MIC.compute_alphas(Ahat, results.mA_opt, WShat, WYhat);
        alpha_hat = nan(size(WShat, 2), 4);
        alpha_hat(:, 1) = abc.in;
        alpha_hat(:, 2) = abc.Srot;
        alpha_hat(:, 3) = abc.Yrot;
        alpha_hat(:, 4) = abc.new;
        alpha_hat_all{kkk, iii} = alpha_hat;

        % constrained MIC fits
        EVcstr = zeros(1, 4);
        params2 = params;
        params2.mA = results.mA_opt; params2.mB = results.mB_opt; params2.mC = results.mC_opt; params2.mD = results.mD_opt;

        params2.opts.source_constraint = "same"; params2.opts.target_constraint = "same";
        abc = MIC.model_wrapper(S, X, Y, params2); % confine source and target MIC to AIS
        EVcstr(1) = abc.ve_mod;

        params2.opts.source_constraint = "none"; params2.opts.target_constraint = "same";
        abc = MIC.model_wrapper(S, X, Y, params2); % confine target MIC to AIS
        EVcstr(2) = abc.ve_mod;

        params2.opts.source_constraint = "same"; params2.opts.target_constraint = "none";
        abc = MIC.model_wrapper(S, X, Y, params2); % confine source MIC to AIS
        EVcstr(3) = abc.ve_mod;

        EVcstr(4) = results.ve_mod(results.mC_try==results.mC_opt); % unconstrained

        EVcstr_all(kkk, iii, :) = EVcstr;
    end
end

% plot results

cols = [0.6 0.7799 0.2420 ; 0.2917 0.0396 0.9867 ; 0.6950 0.4805 0.1425 ; 0.8601 0.1049 0.4989];

figure; % different columns correspnd to different numbers of MICs
for kkk = 1:length(mClist)
    subplot(1, length(mClist), kkk); hold on;
    for iii = 1:num_sims
        est = alpha_hat_all{kkk, iii}; % estimated geometry indices
        gt = nan(size(est)); % ground-truth
        
        gt(:, 1) = cos(thetaS_gt{kkk, iii}).^2 .* cos(thetaY_gt{kkk, iii}).^2;
        gt(:, 2) = sin(thetaS_gt{kkk, iii}).^2 .* cos(thetaY_gt{kkk, iii}).^2;
        gt(:, 3) = cos(thetaS_gt{kkk, iii}).^2 .* sin(thetaY_gt{kkk, iii}).^2;
        gt(:, 4) = sin(thetaS_gt{kkk, iii}).^2 .* sin(thetaY_gt{kkk, iii}).^2;
        
        for j = 1:4
            scatter(gt(:, j), est(:, j), 40, cols(j, :), 'filled', 'o', 'MarkerFaceAlpha', 0.3);
        end
    end
    plot([0, 1], [0, 1], 'k', 'LineWidth', 1.5); % identity line
end

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
