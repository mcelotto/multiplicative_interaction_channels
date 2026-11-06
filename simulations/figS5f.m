%%%%%%%%%%%%%%%%%%%%%%%
% simulations shown in Fig. S5f
% MICs applied to a nonlinear generator: exponential,
% Y = (S*A).^(1 + a*(2X - 1)) (componentwise) + noise, with S, X uniform
% in (0, 1) and A with positive entries (Appendix B.10)
%%%%%%%%%%%%%%%%%%%%%%%

% simulation settings
a_list = linspace(0, 1, 11); % modulation strength parameter
num_runs_per = 20;
N = 1e4;
nS = 5; nY = 5; % nX = 1 always here
mA_gt = 2;
noise = 1;

% fitting setttings
params = struct;
params.max_lin_rank = 10;
params.K = 10;
params.ctgs_folds_on = true; 
params.seed = 42; 
params.opts = struct;
params.stratify_by = [];
params.max_mod_rank = 10;

results_all = cell(length(a_list), num_runs_per);

for jjj = 1:length(a_list)
    for kkk = 1:num_runs_per
        disp([jjj kkk])
        a = a_list(jjj);
        rng("shuffle");


        UA = abs(randn(nS, mA_gt)); VA = abs(randn(nY, mA_gt)); UA = UA./vecnorm(UA, 2, 1); VA = VA./vecnorm(VA, 2, 1);
        A_gt = UA*VA';

        S = rand(N, nS);
        X = rand(N, 1);
        Y = noise*randn(N, nY);
        for i = 1:N
            base = S(i, :)*A_gt;
            p = 1 + a*(2*X(i) - 1);
            Y(i, :) = Y(i, :) + base.^p;
        end

        results = MIC.model_wrapper(S, X, Y, params);

        Ahat = mean(results.cvA{results.mA_try==results.mA_opt}.A, 3);
        WShat = mean(results.cv_mod{results.mC_try==results.mC_opt}.WS, 3);
        WYhat = mean(results.cv_mod{results.mC_try==results.mC_opt}.WY, 3);

        % compute alphas, shuffle subtracted
        if size(WShat, 2)>0
            abc = MIC.compute_alphas(Ahat, results.mA_opt, WShat, WYhat);
            alpha = [abc.in ; abc.Srot ; abc.Yrot ; abc.new];

            % rotation null
            nshufs = 1000; % number of shuffles used to estimate null distribution
            alpha_shuf = nan(nshufs, 4, size(WShat, 2));
            for i = 1:nshufs
                abc = MIC.compute_alphas(Ahat, results.mA_opt, randn(size(WShat)), randn(size(WYhat)));
                alpha_shuf(i, 1, :) = abc.in;
                alpha_shuf(i, 2, :) = abc.Srot;
                alpha_shuf(i, 3, :) = abc.Yrot;
                alpha_shuf(i, 4, :) = abc.new;
            end
            alpha_shuf_mean = squeeze(mean(alpha_shuf, 1));
            if size(alpha_shuf_mean, 1)==1; alpha_shuf_mean = alpha_shuf_mean'; end
            alpha = alpha - alpha_shuf_mean;
        else
            alpha = nan;
        end

        abra = struct;
        abra.results = results;
        abra.alpha_sub = alpha;
        abra.S = S;
        abra.X = X;
        abra.Y = Y;
        abra.A_gt = A_gt;
        results_all{jjj, kkk} = abra;
    end
end

% plot results
ve_mod = nan(size(results_all));
gt_ve_mod = nan(size(results_all));
alpha_sub_avg = nan([size(results_all) 4]);
strengthening_idx = nan(size(results_all));

mCopt_all = [];
for i = 1:size(results_all, 1)
    for j = 1:size(results_all, 2)
        disp([i, j]);
        mCopt_all(i, j) = results_all{i, j}.results.mC_opt;
        curr = results_all{i, j}.results;
        curr2 = results_all{i, j};
        ve_mod(i, j) = curr.ve_mod(curr.mC_try==curr.mC_opt);

        cvp = curr.cv_mod{1}.cv_partition;
        % compute ground-truth model performance
        ve_mod_gt = nan(1, params.K);
        for l = 1:params.K
            tr = training(cvp, l); te = test(cvp, l);
            [Str, Ste] = MIC.split_normalize(curr2.S, tr, te);
            [Xtr, Xte] = MIC.split_normalize(curr2.X, tr, te);
            [Ytr, Yte] = MIC.split_normalize(curr2.Y, tr, te);

            Xte_unn = curr2.X(te, :); Ste_unn = curr2.S(te, :);
            Ytr_unn = curr2.Y(tr, :);
            gt_pred = zeros(size(Yte));
            gt_pred_AIS = Ste_unn*curr2.A_gt;
            for m = 1:size(Yte, 1)
                base = Ste_unn(m, :)*curr2.A_gt;
                p = 1 + a_list(i)*(2*Xte_unn(m) - 1);
                gt_pred(m, :) = base.^p;

            end

            gt_pred = (gt_pred - mean(Ytr_unn, 1))./std(Ytr_unn, [], 1);
            gt_pred_AIS = (gt_pred_AIS - mean(Ytr_unn, 1))./std(Ytr_unn, [], 1);
            ve_mod_gt(l) = MIC.compute_EV(Ytr, Yte, gt_pred) - MIC.compute_EV(Ytr, Yte, gt_pred_AIS);
        end
        gt_ve_mod(i, j) = mean(ve_mod_gt);

        alpha_sub_avg(i, j, :) = mean(curr2.alpha_sub, 2)';


    end
end

mu = mean(ve_mod, 2);
sem = std(ve_mod, [], 2)/sqrt(size(ve_mod, 2));

mu2 = mean(gt_ve_mod, 2);
sem2 = std(gt_ve_mod, [], 2)/sqrt(size(gt_ve_mod, 2));

figure;
subplot(1, 2, 1); hold on;
errorbar(a_list, mu, sem, 'c', 'LineWidth', 1.5);
errorbar(a_list, mu2, sem2, 'k', 'LineWidth', 1.5);
xticks([0, 1/4, 1/2, 3/4, 1]);
xlabel("a");
ylabel("mod EV");
sam_style;


% the four alphas
mu_alpha = squeeze(nanmean(alpha_sub_avg, 2));
sem_alpha = squeeze(nanstd(alpha_sub_avg, [], 2))./squeeze(sum(~isnan(alpha_sub_avg), 2));
cols = [0.6 0.7799 0.2420 ; 0.2917 0.0396 0.9867 ; 0.6950 0.4805 0.1425 ; 0.8601 0.1049 0.4989];

subplot(1, 2, 2); hold on;
for k = 1:4
    errorbar(a_list, squeeze(mu_alpha(:, k)), squeeze(sem_alpha(:, k)), 'Color', cols(k, :), 'LineWidth', 1.5);
end

xticks([0, 1/4, 1/2, 3/4, 1]);
xlabel("a");
ylabel("\alpha (null subtr.)");
yline(0, '--k', 'LineWidth', 1.5, 'Alpha', 1);
sam_style;
legend("ss", "ns", "sn", "nn", 'Orientation', 'horizontal', 'FontSize', 14);
