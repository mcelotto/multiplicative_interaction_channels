%%%%%%%%%%%%%%%%%%%%%%%
% simulations shown in Fig. S5d
% MICs applied to a nonlinear generator: rectifying output,
% Y = ReLU(S*A + a*((S*WS).*(X*WX))*WY') + noise (Appendix B.10)
%%%%%%%%%%%%%%%%%%%%%%%

% simulation settings
a_list = linspace(0, 1, 6); % modulation strength parameter
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
        
        UA = randn(nS, mA_gt); VA = randn(nY, mA_gt); UA = UA./vecnorm(UA, 2, 1); VA = VA./vecnorm(VA, 2, 1);
        A_gt = UA*VA';

        % so alpha_ss = 1/2, alpha_sn = 1/2 is ground-truth
        WS_gt = randvec_angle_to_subspace(UA, 0);
        WY_gt = randvec_angle_to_subspace(VA, pi/4);
        WX_gt = 1;
        WS_gt = WS_gt./vecnorm(WS_gt, 2, 1); WY_gt = WY_gt./vecnorm(WY_gt, 2, 1);

        S = randn(N, nS);
        X = randn(N, 1);
        Y = my_relu(S*A_gt + a*((S*WS_gt).*(X*WX_gt))*WY_gt.') + noise*randn(N, nY);

        results = MIC.model_wrapper(S, X, Y, params);
        
        abra = struct;
        abra.results = results;
        abra.S = S;
        abra.X = X;
        abra.Y = Y;
        abra.A_gt = A_gt;
        abra.WS_gt = WS_gt;
        abra.WY_gt = WY_gt;
        abra.WX_gt = WX_gt;
        results_all{jjj, kkk} = abra;
    end
end

% plot results
ve_mod = nan(size(results_all));
gt_ve_mod = nan(size(results_all));
alpha_sub_avg = zeros([size(results_all) 4]);
alpha_raw_avg = zeros([size(results_all) 4]);


mCopt_all = [];
mAopt_all = [];
for i = 1:size(results_all, 1)
    for j = 1:size(results_all, 2)
        disp([i, j]);
        mCopt_all(i, j) = results_all{i, j}.results.mC_opt;
        mAopt_all(i, j) = results_all{i, j}.results.mA_opt;
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
            gt_pred_AIS = my_relu(Ste_unn*curr2.A_gt); % modulator clamped to baseline (z = 0)
            gt_pred = my_relu(Ste_unn*curr2.A_gt + a_list(i)*((Xte_unn*curr2.WX_gt).*(Ste_unn*curr2.WS_gt))*curr2.WY_gt');

            gt_pred = (gt_pred - mean(Ytr_unn, 1))./std(Ytr_unn, [], 1);
            gt_pred_AIS = (gt_pred_AIS - mean(Ytr_unn, 1))./std(Ytr_unn, [], 1);
            ve_mod_gt(l) = MIC.compute_EV(Ytr, Yte, gt_pred) - MIC.compute_EV(Ytr, Yte, gt_pred_AIS);
        end
        gt_ve_mod(i, j) = mean(ve_mod_gt);

        for ff = 1:params.K
            Ahat = curr.cvA{curr.mA_try==curr.mA_opt}.A(:, :, ff);
            WShat = curr.cv_mod{curr.mC_try==curr.mC_opt}.WS(:, :, ff);
            WXhat = curr.cv_mod{curr.mC_try==curr.mC_opt}.WX(:, :, ff);
            WYhat = curr.cv_mod{curr.mC_try==curr.mC_opt}.WY(:, :, ff);

            % compute alphas, shuffle subtracted
            if size(WShat, 2)>0
                abc = MIC.compute_alphas(Ahat, curr.mA_opt, WShat, WYhat);
                alpha = [abc.in ; abc.Srot ; abc.Yrot ; abc.new];
    
                % rotation null
                if ff == 1
                    nshufs = 1000; % number of shuffles used to estimate null distribution
                    alpha_shuf = nan(nshufs, 4, size(WShat, 2));
                    for ii = 1:nshufs
                        abc = MIC.compute_alphas(Ahat, curr.mA_opt, randn(size(WShat)), randn(size(WYhat)));
                        alpha_shuf(ii, 1, :) = abc.in;
                        alpha_shuf(ii, 2, :) = abc.Srot;
                        alpha_shuf(ii, 3, :) = abc.Yrot;
                        alpha_shuf(ii, 4, :) = abc.new;
                    end
                    alpha_shuf_mean = squeeze(mean(alpha_shuf, 1));
                    if size(alpha_shuf_mean, 1)==1; alpha_shuf_mean = alpha_shuf_mean'; end

                end

                alpha_sub = alpha - alpha_shuf_mean;
            else
                alpha = nan;
                alpha_sub = nan;
            end



            alpha_sub_avg(i, j, :) = squeeze(alpha_sub_avg(i, j, :)) + 1/params.K * mean(alpha_sub, 2);
            alpha_raw_avg(i, j, :) = squeeze(alpha_raw_avg(i, j, :)) + 1/params.K * mean(alpha, 2);
        end

    end
end

mu = mean(ve_mod, 2);
sem = std(ve_mod, [], 2)/sqrt(size(ve_mod, 2));

mu2 = mean(gt_ve_mod, 2);
sem2 = std(gt_ve_mod, [], 2)/sqrt(size(gt_ve_mod, 2));

figure;
subplot(1, 3, 1); hold on;
errorbar(a_list, mu, sem, 'c', 'LineWidth', 1.5);
errorbar(a_list, mu2, sem2, 'k', 'LineWidth', 1.5);
xticks([0, 1/4, 1/2, 3/4, 1]);
xlabel("a");
ylabel("mod EV");
sam_style;

% the four alphas
mu_alpha = squeeze(nanmean(alpha_raw_avg, 2));
sem_alpha = squeeze(nanstd(alpha_raw_avg, [], 2))./sqrt(squeeze(sum(~isnan(alpha_raw_avg), 2)));
cols = [0.6 0.7799 0.2420 ; 0.2917 0.0396 0.9867 ; 0.6950 0.4805 0.1425 ; 0.8601 0.1049 0.4989];

subplot(1, 3, 3); hold on;
for k = 1:4
    errorbar(a_list, squeeze(mu_alpha(:, k)), squeeze(sem_alpha(:, k)), 'Color', cols(k, :), 'LineWidth', 1.5);
end

xticks([0, 1/4, 1/2, 3/4, 1]);
xlabel("a");
ylabel("\alpha (raw)");
yline(0, '--k', 'LineWidth', 1.5, 'Alpha', 1);
sam_style;
legend("ss", "ns", "sn", "nn", 'Orientation', 'horizontal', 'FontSize', 14);

% null-subtracted
mu_alpha = squeeze(nanmean(alpha_sub_avg, 2));
sem_alpha = squeeze(nanstd(alpha_sub_avg, [], 2))./sqrt(squeeze(sum(~isnan(alpha_sub_avg), 2)));
cols = [0.6 0.7799 0.2420 ; 0.2917 0.0396 0.9867 ; 0.6950 0.4805 0.1425 ; 0.8601 0.1049 0.4989];

subplot(1, 3, 2); hold on;
for k = 1:4
    errorbar(a_list, squeeze(mu_alpha(:, k)), squeeze(sem_alpha(:, k)), 'Color', cols(k, :), 'LineWidth', 1.5);
end

xticks([0, 1/4, 1/2, 3/4, 1]);
xlabel("a");
ylabel("\alpha (null-subtracted)");
yline(0, '--k', 'LineWidth', 1.5, 'Alpha', 1);
sam_style;
legend("ss", "ns", "sn", "nn", 'Orientation', 'horizontal', 'FontSize', 14);


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

function x = my_relu(x)
x(x<0) = 0;
end
