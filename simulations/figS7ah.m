%%%%%%%%%%%%%%%%%%%%%%%
% simulations shown in Fig. S5a-h
% validating geometry decomposition in single-nonzero-alpha scenarios
%%%%%%%%%%%%%%%%%%%%%%%

% simulation settings
nSlist = 20; nYlist = 20; nX = 10; % population dimensionalities
gt = {'ss', 'ns', 'sn', 'nn'}; % four ground-truth scenarios
N = 1000; % number of samples per sim
noise = 1; % emission noise
mod_stren = 1; % MIC strength
mA = 2; mC = 1; % ground-truth AIS and MIC ranks
num_sims = 10; % num sims per scenario

% fit settings
params = struct;
params.max_lin_rank = 10;
params.K = 10;
params.ctgs_folds_on = false; 
params.seed = 42; 
params.opts = struct;
params.max_mod_rank = 5;
params.stratify_by = [];

% store results here
alpha_hat_all_all = cell(length(gt), length(nSlist));
alpha_shuf_all_all = cell(length(gt), length(nSlist));
EVcstr_all_all = cell(length(gt), length(nSlist));
for kkk = 1:length(gt)
    for jjj = 1:length(nSlist)
        nS = nSlist(jjj); nY = nYlist(jjj);
        alpha_shuf_all = [];
        alpha_hat_all = [];
        EVcstr_all = [];
        for iii = 1:num_sims
            disp([kkk, jjj, iii]); rng("shuffle");

            % randomly oriented AIS
            UA = randn(nS, mA); UA = UA./vecnorm(UA, 2, 1);
            VA = randn(nY, mA); VA = VA./vecnorm(VA, 2, 1);
            A = UA*VA';

            % orthonormal bases for the AIS and their orthogonal complements
            UAo = orth(UA);
            VAo = orth(VA);
            UAo_perp = null(UAo.');
            VAo_perp = null(VAo.');

            % MIC uniformly drawn from unit sphere in appropriate subspace
            if strcmp(gt{kkk}, 'ss')  
                CS = randn(mA, mC); CY = randn(mA, mC);
                WS = UAo*CS; WY = VAo*CY;
            elseif strcmp(gt{kkk}, 'ns')
                CS = randn(size(UAo_perp, 2), mC); CY = randn(mA, mC);
                WS = UAo_perp*CS; WY = VAo*CY;
            elseif strcmp(gt{kkk}, 'sn')
                CS = randn(mA, mC); CY = randn(size(VAo_perp,2), mC);
                WS = UAo*CS; WY = VAo_perp*CY;
            else
                CS = randn(size(UAo_perp,2), mC); CY = randn(size(VAo_perp,2), mC);
                WS = UAo_perp*CS; WY = VAo_perp*CY;
            end

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

            % geometry indices
            alpha_hat = MIC.compute_alphas(Ahat, results.mA_opt, WShat, WYhat);
            alpha = MIC.compute_alphas(A, mA, WS, WY);
            nshufs = 1000;
            alpha_shuf = nan(nshufs, 4);
            for i = 1:nshufs
                abc = MIC.compute_alphas(Ahat, results.mA_opt, randn(size(WShat)), randn(size(WYhat)));
                alpha_shuf(i, :) = [mean(abc.in) mean(abc.Srot) mean(abc.Yrot) mean(abc.new)]; 
            end

            alpha_shuf_all(iii, :) = mean(alpha_shuf, 1);
            alpha_hat_all(iii, :) = [mean(alpha_hat.in) mean(alpha_hat.Srot) mean(alpha_hat.Yrot) mean(alpha_hat.new)];
            
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


            EVcstr_all(iii, :) = EVcstr;
        end

        alpha_hat_all_all{kkk, jjj} = alpha_hat_all;
        alpha_shuf_all_all{kkk, jjj} = alpha_shuf_all;
        EVcstr_all_all{kkk, jjj} = EVcstr_all;
    end
end

% plot results
% different columns correspond to different ground-truth scenarios

figure;
for i = 1:4
    subplot(2, 4, i); hold on; % geometry indices (raw)
    curr = alpha_hat_all_all{i, 1};
    for j = 1:4
        scatter(j+0.08*randn(size(curr, 1), 1), curr(:, j), 40, 'k', 'filled', 'o', 'MarkerFaceAlpha', 0.3);
        scatter(j, mean(curr(:, j)), 80, 'k', 'filled', 'o');
    end
    subplot(2, 4, 4+i); hold on; % constrained EV
    curr = EVcstr_all_all{i, 1};
    for j = 1:4
        scatter(j+0.08*randn(size(curr, 1), 1), curr(:, j), 40, 'k', 'filled', 'o', 'MarkerFaceAlpha', 0.3);
        scatter(j, mean(curr(:, j)), 80, 'k', 'filled', 'o');
    end
end