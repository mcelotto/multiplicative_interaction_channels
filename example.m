% Example script: MICs analysis on a single ACA-VIP session
%
% Demonstrates the full MICs analysis pipeline on one of the two example
% sessions released as Supplemental Material (sess27, shown in Fig. 4c, or
% sess14, shown in Fig. S6i-k). The script reproduces the core analyses of 
% Fig. 4 and Fig. S6 on individual sessions: 
% model fitting (Algorithm 1), geometric decomposition of MICs
% relative to the baseline AIS, geometry-constrained fitting, and
% visualization of results.
%
% To shorten runtime, ranks are pre-set:
% 
% session 14: params.mA=2, mB=1, mC=1, mD=1 (default)
% session 27: params.mA=3, mB=1, mC=2, mD=1
%
% rather than searched exhaustively. To run the full rank search, replace
% params2 with params (remove the pre-selected rank fields).
%
% Data: two example sessions are provided in ACAxVIP/. The full 28-session
% dataset (~400 MB) will be released upon acceptance (CC BY 4.0).
%
% Usage:
%   1. Set your_path to the repository root
%   2. Run the script; no further inputs required
%
% Outputs (figures, for one session only — paper aggregates results across all 28 sessions):
%   Panel 1 - Scatter of ACA vs. VIP projections onto MIC 1, color-coded
%             by quantile of behavioral state projection (as in Fig. 4c, left)
%   Panel 2 - ACA-VIP correlation as a function of behavioral state
%             quantile (as in Fig. 4c, inset)
%   Panel 3 - Null-subtracted geometry indices (alpha_ss, alpha_ns,
%             alpha_sn, alpha_nn) (as in Fig. 4d, top)
%   Panel 4 - EVmod under geometry constraints: unconstrained, ACA-only,
%             VIP-only, both-AIS (as in Fig. 4d, bottom)
%
% Dependencies: MIC package (+MIC/ on MATLAB path); MATLAB R2025b;
%               Statistics and Machine Learning Toolbox;
%               Parallel Computing Toolbox

clear;
%% load an example session
your_path = @@ your path
load([your_path '/ACAxVIP/sess14']);

%% run main pipeline
% we will use pre-selected ranks to shorten runtime for the purposes of
% this example
S = abc.ACA; % source
Y = abc.VIP; % target
X = [abc.wheel abc.pupil]; % modulator

params = struct;
params.max_lin_rank = 20; % max rank included in searches for ranks of A, B, D
params.K = 10; % number of cross-val folds
params.ctgs_folds_on = true; % temporally contiguous cross-val
params.seed = 42; % random seed
params.opts = struct; % unconstrained fit
params.stratify_by = []; % no stratification
params.max_mod_rank = 10; % max rank included in search for rank of C

% pre-selected ranks
params2 = params;
params2.mA = 2; params2.mB = 1; params2.mC = 1; params2.mD = 1;

results = MIC.model_wrapper(S, X, Y, params2);

%% compute geometry indices
Ahat = mean(results.cvA{results.mA_opt==results.mA_try}.A, 3);
WShat = mean(results.cv_mod{results.mC_opt==results.mC_try}.WS, 3);
WYhat = mean(results.cv_mod{results.mC_opt==results.mC_try}.WY, 3);
alpha = MIC.compute_alphas(Ahat, results.mA_opt, WShat, WYhat);
alpha = [mean(alpha.in) mean(alpha.Srot) mean(alpha.Yrot) mean(alpha.new)]; % mean across channels

% subtract random rotation null
num_shufs = 1e3;
alpha_shuf = nan(1e3, 4);
for n = 1:num_shufs
    temp = MIC.compute_alphas(Ahat, results.mA_opt, randn(size(WShat)), randn(size(WYhat)));
    alpha_shuf(n, :) = [mean(temp.in) mean(temp.Srot) mean(temp.Yrot) mean(temp.new)];
end
alpha_shuf = mean(alpha_shuf, 1);


%% constrained fits
results = MIC.model_wrapper(S, X, Y, params2);

params2 = params;
params2.max_mod_rank = results.mC_opt; % restrict rank(C) to be <= rank for unconstrained fit

% constraint 1: alpha_ss = 1
disp("constraint 1...")
params2.opts.source_constraint = "same"; params2.opts.target_constraint = "same";
cstr1 = MIC.model_wrapper(S, X, Y, params2);

% constraint 2: alpha_sn = 0 = alpha_nn (constrain MIC to baseline AIS in VIP space)
disp("constraint 2...")
params2.opts.source_constraint = "none"; params2.opts.target_constraint = "same";
cstr2 = MIC.model_wrapper(S, X, Y, params2);

% constraint 3: alpha_ns = 0 = alpha_nn (constrain MIC to baseline AIS in ACA space)
disp("constraint 3...")
params2.opts.source_constraint = "same"; params2.opts.target_constraint = "none";
cstr3 = MIC.model_wrapper(S, X, Y, params2);

%% plot results
figure('units','normalized','outerposition',[0 0 1 1]);

subplot(1, 4, 1); hold on; % scatterplot of projection onto one of the MICs
WXhat = mean(results.cv_mod{results.mC_try==results.mC_opt}.WX, 3); % already stored WShat, WYhat
WShat = WShat./vecnorm(WShat, 2, 1); % normalize
WYhat = WYhat./vecnorm(WYhat, 2, 1);
WXhat = WXhat./vecnorm(WXhat, 2, 1);

qq = linspace(0, 1, 7); % quantiles of modulator projection
cc = nan(1, length(qq)-1);
cols = cool(length(qq)-1);

keep = sum(isnan([S X Y]), 2)==0;
S = S(keep, :); X = X(keep, :); Y = Y(keep, :);
MIC_idx = 1; % For sess 27 in Fig. 4c, MIC_idx = 2 was shown
ss = zscore(S, [], 1)*WShat(:, MIC_idx); xx = zscore(X, [], 1)*WXhat(:, MIC_idx); yy = zscore(Y, [], 1)*WYhat(:, MIC_idx); % project onto example MIC
for k = 1:(length(qq)-1)
    if k == (length(qq)-1)
        keep2 = quantile(xx, qq(k))<=xx & xx<=quantile(xx, qq(k+1));
    else
        keep2 = quantile(xx, qq(k))<=xx & xx<quantile(xx, qq(k+1));
    end
    cc(k) = corr(ss(keep2), yy(keep2));

    scatter(ss(keep2), yy(keep2), 15, cols(k, :), 'filled', 'o', 'MarkerFaceAlpha', 0.1, 'MarkerEdgeColor', 'none');
    p = polyfit(ss(keep2), yy(keep2), 1);
    plot([-5 5], [-5 5]*p(1)+p(2), 'Color', cols(k, :), 'LineWidth', 2);
end

xlabel('ACA proj. onto MIC');
ylabel('VIP proj. onto MIC');
c = colorbar; c.LineWidth = 1.5; c.Location = "NorthOutside";
ylabel(c, 'behav. proj. onto MIC (quantile)', 'FontSize', 14);
caxis([0 1]);
colormap('cool');
style();

subplot(1, 4, 2); hold on; % correlation of ACA-VIP projections onto MIC in prev. subplot, as a function of behavior projection onto MIC
qm = (qq(1:(end-1))+qq(2:end))/2;
p = polyfit(qm, cc, 1);
scatter(qm, cc, 80, cols, 'filled', 'o', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 1);
plot(qm, p(1)*qm+p(2), 'k', 'LineWidth', 2); 

xlabel('behav. proj. onto MIC (quantile)');
ylabel('corr(ACA proj., VIP proj.)');
style();

subplot(1, 4, 3); hold on; % shuffle subtracted geometry coefficients
bar(1:4, alpha-alpha_shuf, 0.75, 'LineWidth', 1.5, 'EdgeColor', 'k', 'FaceColor', 'k', 'FaceAlpha', 0.3);
yline(0, 'k', 'LineWidth', 1.5);
xticks(1:4);
xticklabels({'ss', 'ns', 'sn', 'nn'});
xlim([0.2 4.8]);
ylabel('\alpha (null subtr.)');
style();

subplot(1, 4, 4); hold on; % hierarchy of constrained fits
EV_mod_cstr = [results.ve_mod cstr3.ve_mod(cstr3.mC_opt==cstr3.mC_try) ...
    cstr2.ve_mod(cstr2.mC_opt==cstr2.mC_try) cstr1.ve_mod(cstr1.mC_opt==cstr1.mC_try)];
bar(1:4, EV_mod_cstr, 0.75, 'LineWidth', 1.5, 'EdgeColor', 'k', 'FaceColor', 'k', 'FaceAlpha', 0.3);
yline(0, 'k', 'LineWidth', 1.5);
xticks(1:4);
xticklabels({'none', 'ACA', 'VIP', 'both'});
xlim([0.2 4.8]);
ylabel('mod. EV');
style();

set(gcf, 'Color', 'w');

% plotting helper
function style()
    set(gca, 'FontSize', 12);
    set(gca, 'LineWidth', 1.5);
    box off;
    set(gca, 'TickDir', 'out');
    set(gca, 'TickLength', 0.03*[1, 1]);
    set(get(gca,'XLabel'),'FontSize',16);
    set(get(gca,'YLabel'),'FontSize',16);
    axis square;
end
