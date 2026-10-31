%%%%%%%%%%%%%%%%%%%%%%%%%%
% code to produce Fig S7
% robustness of modulation EV and geometry indices 
% to non-convexity of modulation loss function
%%%%%%%%%%%%%%%%%%%%%%%%%%

% fitting parameters
params = struct;
params.max_lin_rank = 20;
params.K = 10;
params.ctgs_folds_on = true; 
params.seed = 42;
params.opts = struct;
params.stratify_by = [];
params.max_mod_rank = 10;

% previously selected ranks
mA = [2 2 2 2 3 2 1 1 2 1 1 1 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 1];
mB = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 0 1 1 1 1 1 1 1];
mC = [1 3 3 0 3 1 0 3 0 1 1 1 3 1 2 2 2 1 1 0 0 2 4 0 2 1 2 0];
mD = [1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1];

% number of choices of ALS fitting seed per session
num_seeds_per = 10;

res = cell(28, num_seeds_per);
for idx = 1:28
    % read data
    [ACA, VIP, wheel, pupil, stim_on, puff, t, type] = readACAVIP(idx);
    disp(size(VIP)); disp(size(ACA))
    X = [wheel pupil];
    S = ACA;
    Y = VIP;

    % pre-selected ranks
    params2 = params;
    params2.mA = mA(idx); params2.mB = mB(idx); params2.mC = mC(idx); params2.mD = mD(idx);

    for l = 1:num_seeds_per
        disp([idx, l]);
        rng("shuffle")
        params2.als_seed = randi(100000); % set ALS fitting seed
        res{idx, l} = MIC.model_wrapper(S, X, Y, params2);
    end
end

% aggregate results
ve_mod = nan(size(res));
alpha_sub = nan([size(res) 4]);
for i = 1:size(res, 1)
    for j = 1:size(res, 2)
        % mod EV
        ve_mod(i, j) = res{i, j}.ve_mod;

        % geometry indices
        if res{i, j}.mC_opt>0
            Ahat = mean(res{i, j}.cvA.A, 3);
            WShat = mean(res{i, j}.cv_mod.WS, 3);
            WYhat = mean(res{i, j}.cv_mod.WY, 3);

            alpha_hat = MIC.compute_alphas(Ahat, res{i, j}.mA_opt, WShat, WYhat);
            nshufs = 1000;
            alpha_shuf = nan(nshufs, 4);
            for k = 1:nshufs
                abc = MIC.compute_alphas(Ahat, res{i, j}.mA_opt, randn(size(WShat)), randn(size(WYhat)));
                alpha_shuf(k, :) = [mean(abc.in) mean(abc.Srot) mean(abc.Yrot) mean(abc.new)]; 
            end
    
            alpha_sub(i, j, :) = [mean(alpha_hat.in) mean(alpha_hat.Srot) mean(alpha_hat.Yrot) mean(alpha_hat.new)] - mean(alpha_shuf, 1);
        end
    end
end

load(@@ path to list of sessions with significant modulation); %-->keep

% plot
figure;
subplot(2, 1, 1); hold on;
for i = 1:size(ve_mod, 2)
    xx = (1:28)+0.08*randn(1, 28);
    scatter(xx, ve_mod(:, i), 40, 'k', 'filled', 'o', 'MarkerFaceAlpha', 0.2); 
end

ylabel('mod. EV', 'FontSize', 16)

cols = [0.6 0.7799 0.2420 ; 0.2917 0.0396 0.9867 ; 0.6950 0.4805 0.1425 ; 0.8601 0.1049 0.4989];

subplot(2, 1, 2); hold on;
for i = 1:size(ve_mod, 2)
    for j = 1:4
        xx = (1:28)+0.08*randn(1, 28); xx = xx(keep);
        scatter(xx, alpha_sub(keep, i, j), 40, cols(j, :), 'filled', 'o', 'MarkerFaceAlpha', 0.2); 
    end
end

ylabel('\alpha (null subtr.)', 'FontSize', 16)
xlabel('session #', 'FontSize', 16)