results_path = @@ your path; % folder where rotation_sim_results_2.mat (used by figS6.m) is saved

% simulation settings
rot_setting_list = {"perp", "par"};
rot_amp_list = linspace(0, pi, 11); % rotation amplitude
num_runs_per = 10;
N = 1e4;
nS = 3; nX = 1; nY = 3;
noise = 1;

% fitting setttings
params = struct;
params.max_lin_rank = 10;
params.K = 10;
params.ctgs_folds_on = true; 
params.seed = 42; 
params.opts = struct;
params.stratify_by = [];
params.max_mod_rank = 5;

results_all = cell(length(rot_setting_list), length(rot_amp_list), num_runs_per);
for iii = 1:length(rot_setting_list)
    rot_setting = rot_setting_list{iii};
    for jjj = 1:length(rot_amp_list)
        for kkk = 1:num_runs_per
            disp([iii jjj kkk])
            rot_amp = rot_amp_list(jjj);
            rng("shuffle");
            if strcmp(rot_setting, "par")
                rot_axis = [randn(2, 1) ; 0]; rot_axis = rot_axis/vecnorm(rot_axis, 2);
            elseif strcmp(rot_setting, "perp")
                rot_axis = [0 ; 0 ; 1];
            else
                disp("PROBLEM: invalid rot setting");
            end

            % A = zeros(3, 3);
            % UAs = randn(2, 2); VAs = randn(2, 2); UAs = UAs./vecnorm(UAs, 2, 1); VAs = VAs./vecnorm(VAs, 2, 1);
            % A(1:2, 1:2) = UAs*VAs';
            A = [1 0 0 ; 0 1 0 ; 0 0 0];



            WX = randn(nX, 1); WX = WX./vecnorm(WX, 2, 1);

            S = randn(N, nS);
            X = randn(N, nX);
            Y = noise*randn(N, nY);
            for i = 1:N
                % rotate in source only
                theta = X(i, :)*WX*rot_amp/3;
                Acurr = rotation_matrix(theta, rot_axis)'*A;
                Y(i, :) = Y(i, :) + S(i, :)*Acurr;
            end

            results = MIC.model_wrapper(S, X, Y, params);

            abra = struct;
            abra.results = results;
            abra.S = S;
            abra.X = X;
            abra.Y = Y;
            abra.A = A;
            abra.WX = WX;
            abra.rot_axis = rot_axis;

            results_all{iii, jjj, kkk} = abra;
        end
    end
end

save(fullfile(results_path, 'rotation_sim_results_2'), "results_all", "-v7.3");

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
        for k = 1:size(results_all, 3)
            mCopt_all(i, j, k) = results_all{i, j, k}.results.mC_opt;
            mAopt_all(i, j, k) = results_all{i, j}.results.mA_opt;
            curr = results_all{i, j, k}.results;
            curr2 = results_all{i, j, k};
            ve_mod(i, j, k) = curr.ve_mod(curr.mC_try==curr.mC_opt);

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
                gt_pred_AIS = Ste_unn*curr2.A;
                for m = 1:size(Yte, 1)
                    theta = Xte_unn(m, :)*curr2.WX*rot_amp_list(j)/3;
                    Acurr = rotation_matrix(theta, curr2.rot_axis)'*curr2.A;
                    gt_pred(m, :) = Ste_unn(m, :)*Acurr;

                end

                gt_pred = (gt_pred - mean(Ytr_unn, 1))./std(Ytr_unn, [], 1);
                gt_pred_AIS = (gt_pred_AIS - mean(Ytr_unn, 1))./std(Ytr_unn, [], 1);
                ve_mod_gt(l) = MIC.compute_EV(Ytr, Yte, gt_pred) - MIC.compute_EV(Ytr, Yte, gt_pred_AIS);
            end
            gt_ve_mod(i, j, k) = mean(ve_mod_gt);

  
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


    
                alpha_sub_avg(i, j, k, :) = squeeze(alpha_sub_avg(i, j, k, :)) + 1/params.K * mean(alpha_sub, 2);
                alpha_raw_avg(i, j, k, :) = squeeze(alpha_raw_avg(i, j, k, :)) + 1/params.K * mean(alpha, 2);
                

            end

        end
    end
end

mu = mean(ve_mod, 3)';
sem = std(ve_mod, [], 3)'/sqrt(size(ve_mod, 3));

mu2 = mean(gt_ve_mod, 3)';
sem2 = std(gt_ve_mod, [], 3)'/sqrt(size(gt_ve_mod, 3));

figure;
subplot(2, 3, 1); hold on;
errorbar(rot_amp_list, mu(:, 1), sem(:, 1), 'c', 'LineWidth', 1.5);
errorbar(rot_amp_list, mu2(:, 1), sem2(:, 1), 'k', 'LineWidth', 1.5);
xticks(pi*[0, 1/4, 1/2, 3/4, 1]);
xticklabels({'0', '\pi/4', '\pi/2', '3\pi/4', '\pi'})
xlabel("rotation amplitude");
ylabel("mod EV");
sam_style;
title("\perp rotation axis", 'FontSize', 16)

subplot(2, 3, 4); hold on;
errorbar(rot_amp_list, mu(:, 2), sem(:, 2), 'c', 'LineWidth', 1.5);
errorbar(rot_amp_list, mu2(:, 2), sem2(:, 2), 'k', 'LineWidth', 1.5);
xticks(pi*[0, 1/4, 1/2, 3/4, 1]);
xticklabels({'0', '\pi/4', '\pi/2', '3\pi/4', '\pi'})
xlabel("rotation amplitude");
ylabel("mod EV");
sam_style;
title("|| rotation axis", 'FontSize', 16);

% the four alphas
% raw indices
mu_alpha = squeeze(nanmean(alpha_raw_avg, 3));
sem_alpha = squeeze(nanstd(alpha_raw_avg, [], 3))./sqrt(squeeze(sum(~isnan(alpha_raw_avg), 3)));
cols = [0.6 0.7799 0.2420 ; 0.2917 0.0396 0.9867 ; 0.6950 0.4805 0.1425 ; 0.8601 0.1049 0.4989];

subplot(2, 3, 3); hold on;
for k = 1:4
    errorbar(rot_amp_list, squeeze(mu_alpha(1, :, k)), squeeze(sem_alpha(1, :, k)), 'Color', cols(k, :), 'LineWidth', 1.5);
end

xticks(pi*[0, 1/4, 1/2, 3/4, 1]);
xticklabels({'0', '\pi/4', '\pi/2', '3\pi/4', '\pi'})
xlabel("rotation amplitude");
ylabel("\alpha (null subtr.)");
yline(0, '--k', 'LineWidth', 1.5, 'Alpha', 1);
sam_style;
legend("ss", "ns", "sn", "nn", 'Orientation', 'horizontal', 'FontSize', 14);

subplot(2, 3, 6); hold on;
for k = 1:4
    errorbar(rot_amp_list, squeeze(mu_alpha(2, :, k)), squeeze(sem_alpha(2, :, k)), 'Color', cols(k, :), 'LineWidth', 1.5);
end
xticks(pi*[0, 1/4, 1/2, 3/4, 1]);
xticklabels({'0', '\pi/4', '\pi/2', '3\pi/4', '\pi'})
xlabel("rotation amplitude");
ylabel("\alpha (null subtr.)");
yline(0, '--k', 'LineWidth', 1.5, 'Alpha', 1);
sam_style;

% null-subtracted indices
mu_alpha = squeeze(nanmean(alpha_sub_avg, 3));
sem_alpha = squeeze(nanstd(alpha_sub_avg, [], 3))./sqrt(squeeze(sum(~isnan(alpha_sub_avg), 3)));
cols = [0.6 0.7799 0.2420 ; 0.2917 0.0396 0.9867 ; 0.6950 0.4805 0.1425 ; 0.8601 0.1049 0.4989];

subplot(2, 3, 2); hold on;
for k = 1:4
    errorbar(rot_amp_list, squeeze(mu_alpha(1, :, k)), squeeze(sem_alpha(1, :, k)), 'Color', cols(k, :), 'LineWidth', 1.5);
end

xticks(pi*[0, 1/4, 1/2, 3/4, 1]);
xticklabels({'0', '\pi/4', '\pi/2', '3\pi/4', '\pi'})
xlabel("rotation amplitude");
ylabel("\alpha (null subtr.)");
yline(0, '--k', 'LineWidth', 1.5, 'Alpha', 1);
sam_style;
legend("ss", "ns", "sn", "nn", 'Orientation', 'horizontal', 'FontSize', 14);

subplot(2, 3, 5); hold on;
for k = 1:4
    errorbar(rot_amp_list, squeeze(mu_alpha(2, :, k)), squeeze(sem_alpha(2, :, k)), 'Color', cols(k, :), 'LineWidth', 1.5);
end
xticks(pi*[0, 1/4, 1/2, 3/4, 1]);
xticklabels({'0', '\pi/4', '\pi/2', '3\pi/4', '\pi'})
xlabel("rotation amplitude");
ylabel("\alpha (null subtr.)");
yline(0, '--k', 'LineWidth', 1.5, 'Alpha', 1);
sam_style;

function R = rotation_matrix(theta, axis)
%ROTATION_MATRIX 3D rotation matrix from axis-angle representation
%
% Inputs:
%   theta : rotation angle (radians)
%   axis  : 3D rotation axis [x; y; z] or [x, y, z]
%
% Output:
%   R     : 3x3 rotation matrix (convention: acts to the right on column
%   vectors)

    % Ensure column vector
    axis = axis(:);

    % Normalize axis
    axis = axis / norm(axis);

    ux = axis(1);
    uy = axis(2);
    uz = axis(3);

    c = cos(theta);
    s = sin(theta);
    C = 1 - c;

    R = [
        c + ux^2*C,      ux*uy*C - uz*s,  ux*uz*C + uy*s;
        uy*ux*C + uz*s, c + uy^2*C,       uy*uz*C - ux*s;
        uz*ux*C - uy*s, uz*uy*C + ux*s,  c + uz^2*C
    ];

end
