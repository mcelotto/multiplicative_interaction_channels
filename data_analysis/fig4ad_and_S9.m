%%%%%%%%%%%%%%%%%%%%%%%%
% code to generate fig 4a-d and fig S9a-d
% stimulus modulation of V1-V2 interactions (data of Zandvakili & Kohn
% 2015, analyzed in Semedo et al. 2019)
%
% Run the fitting section twice: with params.stim_code = 'one_hot' (all
% panels) and with params.stim_code = '2dim' (periodic encoding, used
% only in fig S9d). The bilinearity diagnostic (fig S9e) is in
% figS9e_and_S11g.m
%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

save_path = @@ your path; % folder for fitting results
fig_save_path = @@ your path; % folder for figures

% fitting settings
params = struct;
params.max_lin_rank = 20;
params.K = 10;
params.ctgs_folds_on = true;
params.seed = 42;
params.opts = struct;
params.stratify_by = [];
params.max_mod_rank = 10;
params.tbin = 100; % spike count bin width (ms)
params.stim_code = 'one_hot'; % 'one_hot': one-hot encoding of the 8 orientations; '2dim': (sin(2*theta), cos(2*theta))

% other settings
Kouter = 2; nshuffle = 2e3; shuffle_type = "circshift"; % shuffle test settings
sig_level = 0.05;
nsess = 5;

% store results here
res = cell(nsess, 1); % main pipeline
res_cstr = cell(nsess, 4); % constrained fits
res_shuf = cell(nsess, 1); % complete output of the shuffle test
p_A = nan(nsess, 1); p_mod = nan(nsess, 1); % AIS and MIC p-values
res_nn = cell(nsess, 1); % nn-only fit with its own rank search (fig S9c)
res_nn_shuf = cell(nsess, 1); % shuffle test of the nn-only fit
p_A_nn = nan(nsess, 1); p_mod_nn = nan(nsess, 1);

% run main pipeline, constrained fits, and significance tests on all
% sessions
for idx = 1:nsess
    disp(idx)
    [V1, V2, stim] = readSemedo(idx, params); % read data
    disp(size(V1)); disp(size(V2))
    S = V1'; % source
    X = stim'; % modulator
    Y = V2'; % target

    res{idx} = MIC.model_wrapper(S, X, Y, params);

    if res{idx}.mC_opt>0
        % constrained fits (re-use additive term ranks, re-select mod rank but
        % require it to be <= mod rank for unconstrained fit)
        params2 = params;
        params2.max_mod_rank = res{idx}.mC_opt;
        params2.mA = res{idx}.mA_opt; params2.mB = res{idx}.mB_opt; params2.mD = res{idx}.mD_opt;

        params2.opts.source_constraint = "same"; params2.opts.target_constraint = "same";
        res_cstr{idx, 1} = MIC.model_wrapper(S, X, Y, params2); % MIC confined to AIS in source and target (both)

        params2.opts.source_constraint = "none"; params2.opts.target_constraint = "same";
        res_cstr{idx, 2} = MIC.model_wrapper(S, X, Y, params2); % MIC confined to AIS in target (V2)

        params2.opts.source_constraint = "same"; params2.opts.target_constraint = "none";
        res_cstr{idx, 3} = MIC.model_wrapper(S, X, Y, params2); % MIC confined to AIS in source (V1)

        params2.opts.source_constraint = "new"; params2.opts.target_constraint = "new";
        res_cstr{idx, 4} = MIC.model_wrapper(S, X, Y, params2); % MIC confined outside AIS in source and target (nn only)
    end

    % run shuffle test to assess significance of AIS and MIC
    [inner_model, EV_A, EV_B, EV_D, EV_mod, EV_A_shuf, EV_B_shuf, EV_D_shuf, EV_mod_shuf, p_A(idx), p_B, p_D, p_mod(idx)] = MIC.heldout_test(S, X, Y, params, Kouter, nshuffle, shuffle_type);
    abc = struct; abc.inner_model = inner_model; abc.EV_A = EV_A; abc.EV_mod = EV_mod; abc.EV_A_shuf = EV_A_shuf; abc.EV_mod_shuf = EV_mod_shuf; abc.p_A = p_A; abc.p_mod = p_mod;
    res_shuf{idx} = abc;

    % nn-only fit (ranks re-selected, mod rank <= mod rank for unconstrained
    % fit) and its shuffle test (fig S9c)
    if res{idx}.mC_opt>0
        params_nn = params;
        params_nn.max_mod_rank = res{idx}.mC_opt;
        params_nn.opts.source_constraint = "new"; params_nn.opts.target_constraint = "new";
        res_nn{idx} = MIC.model_wrapper(S, X, Y, params_nn);

        [inner_model, EV_A, EV_B, EV_D, EV_mod, EV_A_shuf, EV_B_shuf, EV_D_shuf, EV_mod_shuf, p_A_nn(idx), p_B, p_D, p_mod_nn(idx)] = MIC.heldout_test_cstr(S, X, Y, params_nn, Kouter, nshuffle, shuffle_type);
        abc = struct; abc.inner_model = inner_model; abc.EV_A = EV_A; abc.EV_mod = EV_mod; abc.EV_A_shuf = EV_A_shuf; abc.EV_mod_shuf = EV_mod_shuf; abc.p_A = p_A_nn; abc.p_mod = p_mod_nn;
        res_nn_shuf{idx} = abc;
    end
end

% identify sessions with significant modulation
mC = nan(1, nsess);
for i = 1:nsess
    mC(i) = res{i}.mC_opt*(p_mod(i)<sig_level);
end
keep = mC>0;

% save
save(fullfile(save_path, [params.stim_code '_sig_heldout_V1_V2']), "res", "res_cstr", "res_shuf", "p_A", "p_mod", ...
    "res_nn", "res_nn_shuf", "p_A_nn", "p_mod_nn", "keep", "params");

%% geometry indices and modulation EV of constrained fits (one-hot encoding)
rng(0);
load(fullfile(save_path, 'one_hot_sig_heldout_V1_V2'));
d2 = load(fullfile(save_path, '2dim_sig_heldout_V1_V2'), "res"); % periodic encoding (fig S9d)
nsess = numel(res);
animal_id = [1 2 2 3 3]; % animal of each session, in readSemedo order

mC = nan(1, nsess); mA = nan(1, nsess);
for i = 1:nsess
    mC(i) = res{i}.mC_opt;
    mA(i) = res{i}.mA_opt;
end

% modulation EV at the selected rank: unconstrained fit, constrained fits,
% and unconstrained fit with periodic encoding
EVmod = nan(nsess, 1);
EVmod_cstr = nan(nsess, 3); % columns: V1 (source in AIS), V2 (target in AIS), both (source and target in AIS)
EVmod_2d = nan(nsess, 1);
for i = 1:nsess
    EVmod(i) = ve_at_opt(res{i});
    EVmod_2d(i) = ve_at_opt(d2.res{i});
    if res{i}.mC_opt>0
        EVmod_cstr(i, 1) = ve_at_opt(res_cstr{i, 3});
        EVmod_cstr(i, 2) = ve_at_opt(res_cstr{i, 2});
        EVmod_cstr(i, 3) = ve_at_opt(res_cstr{i, 1});
    end
end

% geometry indices, computed in each CV fold (A, WS, WY fitted on the same
% training data) and averaged across folds
nrot = 1e3; % number of random rotations used to estimate null for geometry indices
alpha_hat = nan(nsess, params.K, 4);
alpha_shuf = nan(nsess, params.K, 4);
for i = 1:nsess
    for k = 1:params.K
        Ahat = res{i}.cvA{res{i}.mA_try==res{i}.mA_opt}.A(:, :, k);
        WShat = res{i}.cv_mod{res{i}.mC_try==res{i}.mC_opt}.WS(:, :, k);
        WYhat = res{i}.cv_mod{res{i}.mC_try==res{i}.mC_opt}.WY(:, :, k);
        def = MIC.compute_alphas(Ahat, res{i}.mA_opt, WShat, WYhat);
        if isfield(def, "in")
            alpha_hat(i, k, :) = [mean(def.in) mean(def.Srot) mean(def.Yrot) mean(def.new)];

            % random rotation null
            temp = zeros(1, 4);
            for j = 1:nrot
                WShat = randn(size(WShat));
                WYhat = randn(size(WYhat));
                def = MIC.compute_alphas(Ahat, res{i}.mA_opt, WShat, WYhat);
                temp = temp + [mean(def.in) mean(def.Srot) mean(def.Yrot) mean(def.new)]/nrot;
            end
            alpha_shuf(i, k, :) = temp;
        end
    end
end
% null-subtracted indices of sessions with significant modulation (columns: ss, ns, sn, nn)
alpha_null_subtr = reshape(mean(alpha_hat(keep, :, :), 2) - mean(alpha_shuf(keep, :, :), 2), [], 4);

%% fig 4b and S9b: modulation EV and AIS EV vs. rank
main_fig = figure('Position', [6, 242, 981, 329]);

% modulation EV relative to AIS EV vs. rank (fig 4b)
abc = [];
for i = 1:nsess
    abc(i, :) = [(res{i}.ve_mod(:)')./(res{i}.veA(mA(i)+1))' ...
        res{i}.ve_mod(end)*ones(1, params.max_mod_rank+1-length(res{i}.ve_mod))];
end
col_insig = rand(1, 3);

subplot(2, 14, [1:4 15:18]); hold on;
plot(0:params.max_mod_rank, abc(keep, :), 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
if sum(~keep)
    plot(0:params.max_mod_rank, abc(~keep, :), 'Color', col_insig, 'LineWidth', 1.5);
end
plot(0:params.max_mod_rank, nanmean(abc(keep, :), 1), 'Color', 'r', 'LineWidth', 2.5);
for i = 1:nsess
    if keep(i)
        scatter(mC(i), abc(i, mC(i)+1), 60, 'k', 'filled');
    end
end
ylabel('EV mod / EV AIS')
xlabel('mC')
sam_style

% AIS EV vs. rank (fig S9b)
abc = [];
for i = 1:nsess
    abc(i, :) = [res{i}.veA(:)' res{i}.veA(end)*ones(1, params.max_lin_rank+1-length(res{i}.veA))];
end
col_insig = rand(1, 3);

subplot(2, 14, [6:8 21:23]); hold on;
plot(0:params.max_lin_rank, abc(keep, :), 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
if sum(~keep)
    plot(0:params.max_lin_rank, abc(~keep, :), 'Color', col_insig, 'LineWidth', 1.5);
end
plot(0:params.max_lin_rank, nanmean(abc(keep, :), 1), 'Color', 'r', 'LineWidth', 2.5);
for i = 1:nsess
    if keep(i)
        scatter(mC(i), abc(i, mA(i)+1), 60, 'k', 'filled');
    end
end
ylabel('EV AIS')
xlabel('mA')
sam_style

figure_save(fig_save_path, 'Semedo_ranks', main_fig);

%% fig 4c: V1-V2 score relationship across grating orientations
% left: source vs target MIC scores of an example session, one colour per
%       orientation, orientations ranked by their modulator score
% inset: correlation between source and target scores within each
%        orientation vs modulator score (min-max normalized to [0, 1])
% right: slope of the inset relationship, averaged across MICs, per session
sel_sess = 2; % example session (must satisfy keep(sel_sess))
nOri = 8;
ori_list = 0:22.5:157.5; % stimulus orientations (deg)
cols = cool(nOri); % one colour per score rank

% example session: projections
[src_proj, targ_proj, mod_proj, ori_idx] = mic_scores(res{sel_sess}, sel_sess, params);
nChan = size(src_proj, 2);

% modulator score per orientation (identical for all samples of an orientation)
ori_score = nan(nOri, nChan);
for k = 1:nOri
    ori_score(k, :) = mean(mod_proj(ori_idx == k, :), 1);
end
ori_score = (ori_score - min(ori_score, [], 1)) ./ (max(ori_score, [], 1) - min(ori_score, [], 1));

% scatter + per-orientation regression lines, one subplot per MIC
fig = figure('Position', [195, 47, 476, 693]);
corr_ori = nan(nOri, nChan); % indexed by orientation
ord_all = nan(nOri, nChan); % orientation indices sorted by score, per MIC
for m = 1:nChan
    [~, ord] = sort(ori_score(:, m), 'ascend');
    ord_all(:, m) = ord;
    subplot(ceil(nChan/2), 2, m); hold on
    xl = [min(src_proj(:, m)) max(src_proj(:, m))];
    for kk = 1:nOri % kk = rank, k = orientation index
        k = ord(kk);
        idx = ori_idx == k;
        scatter(src_proj(idx, m), targ_proj(idx, m), 10, 'filled', ...
            'MarkerFaceColor', cols(kk, :), 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.3)
        corr_ori(k, m) = corr(src_proj(idx, m), targ_proj(idx, m));
        c = polyfit(src_proj(idx, m), targ_proj(idx, m), 1);
        plot(xl, polyval(c, xl), 'Color', cols(kk, :), 'LineWidth', 2)
    end
    xlim(xl); ylim([min(targ_proj(:, m)) max(targ_proj(:, m))])

    % colorbar: angular difference from the top-ranked orientation
    colormap(gca, cool(nOri)); caxis([0 1]); cb = colorbar;
    cb.Ticks = ((1:nOri) - 0.5) / nOri;
    th_sorted = ori_list(ord);
    th_pref = th_sorted(end);
    dth = mod(th_sorted - th_pref + 90, 180) - 90; % signed angular difference in (-90, 90]
    cb.TickLabels = compose('%+g°', dth);
    cb.TickLabels{end} = '0°';
    cb.Label.String = sprintf('\\Delta\\theta from %g° (score rank)', th_pref);
    cb.FontSize = 8;
    title(['MIC ', num2str(m)])
    xlabel('V1 score'); ylabel('V2 score')
    sam_style
end
figure_save(fig_save_path, ['Semedo_sess', num2str(sel_sess), '_score_relation'], fig);

% correlation vs modulator score, one column per MIC (top: correlation;
% bottom: score per orientation, in the same colours)
fig = figure('Position', [360, 300, 223*nChan, 380]);
for m = 1:nChan
    ord = ord_all(:, m);

    subplot(2, nChan, m); hold on
    scatter(ori_score(ord, m), corr_ori(ord, m), 40, cols, 'filled')
    c = polyfit(ori_score(:, m), corr_ori(:, m), 1);
    xl = [min(ori_score(:, m)) max(ori_score(:, m))];
    plot(xl, polyval(c, xl), 'k', 'LineWidth', 2)
    text(xl(1), max(corr_ori(:, m)), ['slope = ', num2str(c(1), 2)], 'VerticalAlignment', 'top')
    xlabel('grating score'); ylabel('V1–V2 score corr.')
    title(['MIC ', num2str(m)])
    sam_style

    subplot(2, nChan, nChan + m); hold on
    b = bar(ori_list, ori_score(:, m), 'FaceColor', 'flat', 'EdgeColor', 'none');
    rank_of = nan(nOri, 1);
    rank_of(ord) = 1:nOri; % rank of each orientation
    b.CData = cols(rank_of, :);
    xticks(ori_list); xtickangle(45)
    xlabel('orientation (deg)'); ylabel('grating score')
    sam_style
end
figure_save(fig_save_path, ['Semedo_sess', num2str(sel_sess), '_correlation_vs_score'], fig);

% all sessions: slope of correlation vs modulator score, averaged across MICs
slope_sess = nan(nsess, 1);
for i = find(keep)
    [sp, tp, mp, oi] = mic_scores(res{i}, i, params);
    nC = size(sp, 2);
    slope_chan = nan(1, nC);
    for m = 1:nC
        sc = nan(nOri, 1); cc = nan(nOri, 1);
        for k = 1:nOri
            idx = oi == k;
            sc(k) = mean(mp(idx, m));
            cc(k) = corr(sp(idx, m), tp(idx, m));
        end
        sc = (sc - min(sc)) / (max(sc) - min(sc));
        c = polyfit(sc, cc, 1);
        slope_chan(m) = c(1);
    end
    slope_sess(i) = mean(slope_chan);
end
slope_sess = slope_sess(keep);

fig = figure('Position', [157, 322, 167, 290]); hold on
x_shift = linspace(-0.1, 0.1, numel(slope_sess));
scatter(1 + x_shift, slope_sess, 40, 'filled', 'MarkerFaceColor', [0.5 0.5 0.5], 'MarkerEdgeColor', 'none')
scatter(1, mean(slope_sess), 80, 'r', 'filled')
errorbar(1, mean(slope_sess), std(slope_sess) / sqrt(numel(slope_sess)), 'k', 'LineStyle', 'none', 'LineWidth', 1.5)
yline(0, '--k')
[~, p] = ttest(slope_sess);
pvalues_plot(p, 1, max(slope_sess), 0.1 * range([0; slope_sess]), 1, 14, 0, 'k', 0)
xlim([0.5 1.5]); xticks([])
ylabel('V1–V2 score corr slope')
sam_style
figure_save(fig_save_path, 'Semedo_score_corr_slopes', fig);

%% fig S9a: shuffle test of AIS and modulation EV
ms = nan(size(res_shuf{1}.EV_mod_shuf, 1), nsess);
as = nan(size(ms));
EV_A = nan(1, nsess); EV_mod = nan(1, nsess);
for i = 1:nsess
    ms(:, i) = (res_shuf{i}.EV_mod_shuf(:, 1)+res_shuf{i}.EV_mod_shuf(:, 2))/2; % null distribution of fold-avg mod EV
    as(:, i) = (res_shuf{i}.EV_A_shuf(:, 1)+res_shuf{i}.EV_A_shuf(:, 2))/2; % null distribution of fold-avg AIS EV
    EV_A(i) = mean(res_shuf{i}.EV_A);
    EV_mod(i) = mean(res_shuf{i}.EV_mod);
end

fig_sig = figure('Position', [2363, 173, 399, 422]);
subplot(2, 1, 2); hold on; % mod
boxplot(ms, 'Symbol', ''); % null
scatter(1:nsess, EV_mod); % actual
ylim([-inf, 0.02])
ylabel('mod EV')
title('EV mod significance (one-hot)')
xlabel('Session ID')
sam_style

subplot(2, 1, 1); hold on; % AIS
boxplot(as, 'Symbol', ''); % null
scatter(1:nsess, EV_A); % actual
ylim([-inf, 0.1])
ylabel('EV AIS')
title('EV AIS significance (one-hot)')
sam_style

figure_save(fig_save_path, 'Semedo_significance', fig_sig);

%% fig S9c: shuffle test of the nn-only modulation EV
ms = nan(size(res_nn_shuf{1}.EV_mod_shuf, 1), nsess);
EV_mod = nan(1, nsess);
for i = 1:nsess
    ms(:, i) = (res_nn_shuf{i}.EV_mod_shuf(:, 1)+res_nn_shuf{i}.EV_mod_shuf(:, 2))/2; % null distribution of fold-avg mod EV
    EV_mod(i) = mean(res_nn_shuf{i}.EV_mod);
end

fig = figure('Position', [109, 118, 512, 397]); hold on;
boxplot(ms, 'Symbol', ''); % null
scatter(1:nsess, EV_mod); % actual
ylim([-inf, 0.01])
ylabel('mod EV')
title('EV mod (\alpha_{nn} only) significance')
xlabel('Session ID')
sam_style
figure_save(fig_save_path, 'Semedo_significance_nn_only', fig);

%% fig S9d: periodic vs one-hot encoding of orientation
fig = figure('Position', [2086, 283, 220, 442]); hold on
tmp = EVmod_2d./EVmod;
xshift = linspace(-0.1, 0.1, size(tmp, 1));
scatter(1+xshift, tmp, 'MarkerFaceColor', [0.5, 0.5, 0.5], 'MarkerEdgeColor', [0, 0, 0])
scatter(1, mean(tmp), 'MarkerFaceColor', 'r')
errorbar(1, mean(tmp), std(tmp)/sqrt(nsess), 'LineStyle', 'none', 'Color', 'k')
ylim([0, 1])
ylabel('EV mod (periodic / one-hot)')
xticks([])
sam_style
figure_save(fig_save_path, 'Semedo_periodic_vs_one_hot', fig);

%% fig 4d: constrained fits and geometry indices, LME statistics
% per-session values modeled as d ~ 1 + (1|animal), intercept tested
% against zero (two-tailed); Holm correction across the 3 constrained-fit
% comparisons and across the 4 geometry indices (Appendix B.8, B.14)
animal = categorical(animal_id(keep)');
uA = categories(animal);
nA = numel(uA);

% constrained fits: differences in modulation EV between pairs of fits
cond_names = {'V1', 'V2', 'both'};
pairs = [2 3; 1 3; 1 2]; % V2 vs both (ns), V1 vs both (sn), V1 vs V2
M = EVmod_cstr(keep, :);
stat_cstr = lme_test(M(:, pairs(:, 2)) - M(:, pairs(:, 1)), animal);
stat_cstr.p_holm = holm_adjust(stat_cstr.p_raw);
stat_cstr.Properties.RowNames = strcat(cond_names(pairs(:, 1))', {' vs '}, cond_names(pairs(:, 2))');
disp('--- constrained fits (EV mod), LME ---'); disp(stat_cstr)

% geometry indices: null-subtracted index vs 0
idx_names = {'ss', 'ns', 'sn', 'nn'};
stat_alpha = lme_test(alpha_null_subtr, animal);
stat_alpha.p_holm = holm_adjust(stat_alpha.p_raw);
stat_alpha.Properties.RowNames = idx_names;
disp('--- geometry indices (null subtracted), LME ---'); disp(stat_alpha)

% plot: dots = animal means
fig_lme = figure;

subplot(1, 2, 1); hold on; % constrained fits; red: mean +- SEM across sessions
M_anim = nan(nA, 3);
for a = 1:nA
    M_anim(a, :) = mean(M(animal == uA{a}, :), 1, 'omitnan');
end
for i = 1:3
    scatter(i + 0.05*randn(nA, 1), M_anim(:, i), 55, 'k', 'filled', 'o', 'MarkerFaceAlpha', 0.85);
end
scatter(1:3, mean(M, 1, 'omitnan'), 80, 'r', 'filled', 'o');
errorbar(1:3, mean(M, 1, 'omitnan'), std(M, [], 1, 'omitnan')./sqrt(sum(~isnan(M), 1)), ...
    'k', 'LineStyle', 'none', 'LineWidth', 1.5);
yTop = max(M(:), [], 'omitnan');
for k = 1:size(pairs, 1)
    pvalues_plot(stat_cstr.p_holm(k), mean(pairs(k, :)), yTop + k*(yTop/8), yTop/10, 1, 15, diff(pairs(k, :))/2, 'k', 1);
end
xticks(1:3)
xticklabels(cond_names)
ylabel('mod EV')
title('constrained fits')
sam_style; set(gca, 'TickLength', 0.02*[1, 1]);

subplot(1, 2, 2); hold on; % geometry indices; red: mean +- SEM across animals
alpha_anim = nan(nA, 4);
for a = 1:nA
    alpha_anim(a, :) = mean(alpha_null_subtr(animal == uA{a}, :), 1, 'omitnan');
end
for i = 1:4
    scatter(i + 0.05*randn(nA, 1), alpha_anim(:, i), 40, 'filled', 'o', 'MarkerFaceColor', 'k', 'MarkerFaceAlpha', 0.3);
end
scatter(1:4, mean(alpha_anim, 1), 80, 'r', 'filled', 'o');
yline(0, '--r', 'LineWidth', 1.5);
errorbar(1:4, mean(alpha_anim, 1), std(alpha_anim, [], 1)/sqrt(nA), 'k', 'LineStyle', 'none', 'LineWidth', 1);
for i = 1:4
    pvalues_plot(stat_alpha.p_holm(i), i, 0.8, 0.05, 1, 14, 0, 'k', 1)
end
xticks(1:4)
xticklabels(idx_names)
ylabel('alpha (null subtracted)')
title('geometry indices')
sam_style; set(gca, 'TickLength', 0.02*[1, 1]);

figure_save(fig_save_path, 'Semedo_LME_cstr_geometry', fig_lme);

%%%%%%% HELPERS %%%%%%
function ve = ve_at_opt(r)
% cross-validated modulation EV at the selected modulation rank
if isempty(r.mC_opt)
    ve = NaN;
else
    ve = r.ve_mod(r.mC_opt==r.mC_try);
end
end

function stat = lme_test(D, animal)
% for each column of D (one value per session), fits d ~ 1 + (1|animal)
% and tests the intercept against zero
n = size(D, 2);
est = nan(n, 1); se = nan(n, 1); df = nan(n, 1); p_raw = nan(n, 1);
for i = 1:n
    ok = ~isnan(D(:, i));
    tbl = table(D(ok, i), animal(ok), 'VariableNames', {'d', 'animal'});
    mdl = fitlme(tbl, 'd ~ 1 + (1|animal)');
    est(i) = mdl.Coefficients.Estimate(1);
    se(i) = mdl.Coefficients.SE(1);
    df(i) = mdl.Coefficients.DF(1);
    p_raw(i) = mdl.Coefficients.pValue(1);
end
stat = table(est, se, df, p_raw);
end

function p_adj = holm_adjust(p)
% Holm step-down adjusted p-values
p = p(:); k = numel(p);
[ps, ord] = sort(p, 'ascend');
adj = min(1, ps .* (k:-1:1)'); % p_(j) * (k - j + 1)
adj = cummax(adj); % monotonicity in the sorted order
p_adj = nan(size(p));
p_adj(ord) = adj;
end

function [src_proj, targ_proj, mod_proj, ori_idx] = mic_scores(r, idx, params)
% projections of a session's data onto the fitted MICs (fold-averaged
% weights), after removing the additive terms
[V1, V2, stim] = readSemedo(idx, params);
X = zscore(stim'); % modulator (one-hot orientations, standardized)
S = zscore(V1'); % source
Y = zscore(V2'); % target
[~, ori_idx] = max(stim, [], 1);
ori_idx = ori_idx(:);

sel = r.mC_try == r.mC_opt;
WS = mean(r.cv_mod{sel}.WS, 3); WY = mean(r.cv_mod{sel}.WY, 3); WX = mean(r.cv_mod{sel}.WX, 3);
WS = WS ./ vecnorm(WS, 2, 1); WY = WY ./ vecnorm(WY, 2, 1); WX = WX ./ vecnorm(WX, 2, 1);

% residualize: S_res = S - X*D; Y_res = Y - X*B - S_res*A
A = mean(r.cvA{r.mA_try == r.mA_opt}.A, 3);
if r.mB_opt > 0
    B = mean(r.cvB{r.mB_try == r.mB_opt}.B, 3);
else
    B = zeros(size(X, 2), size(Y, 2));
end
if r.mD_opt > 0
    D = mean(r.cvD{r.mD_try == r.mD_opt}.D, 3);
else
    D = zeros(size(X, 2), size(S, 2));
end
S = S - X * D;
Y = Y - X * B - S * A;

src_proj = S * WS;
targ_proj = Y * WY;
mod_proj = X * WX;
end

function pvalues_plot(pval, pos, maxY, dpt, scale, f2, width, color, star)

if nargin<7
    width = 0.1;
    color = 'k';
    star = 1;
elseif nargin<8
    color = 'k';
    star = 1;
elseif nargin < 9
    star = 1;
end

if maxY > 0
    if star
        if pval > 0.05
            text(pos, maxY+scale*dpt, sprintf('%.1g', pval), 'color', color, 'horizontalalignment', 'center', 'FontSize', f2-4)
        elseif (pval <= 0.05 && pval > 0.01)
            text(pos, maxY+scale*dpt, sprintf('*'), 'color', color, 'horizontalalignment', 'center', 'FontSize', f2)
        elseif (pval <= 0.01 && pval > 0.001)
            text(pos, maxY+scale*dpt, sprintf('**'), 'color', color, 'horizontalalignment', 'center', 'FontSize', f2)
        elseif pval <= 0.001
            text(pos, maxY+scale*dpt, sprintf('***'), 'color', color, 'horizontalalignment', 'center', 'FontSize', f2)
        end
    else
        text(pos, maxY+scale*dpt, sprintf('%.2g', pval), 'color', color, 'horizontalalignment', 'center', 'FontSize', f2-4)
    end

    plot([pos-width, pos+width], [maxY+(scale-.5)*dpt, maxY+(scale-.5)*dpt], ['-', color], 'LineWidth', 1)

elseif maxY < 0
    dpt = -dpt;
    if star
        if pval > 0.055
            text(pos, maxY+scale*dpt, sprintf('%.1g', pval), 'color', color, 'horizontalalignment', 'center', 'FontSize', f2-4)
        elseif (pval < 0.055 && pval > 0.01)
            text(pos, maxY+scale*dpt, sprintf('*'), 'color', color, 'horizontalalignment', 'center', 'FontSize', f2)
        elseif (pval < 0.01 && pval > 0.001)
            text(pos, maxY+scale*dpt, sprintf('**'), 'color', color, 'horizontalalignment', 'center', 'FontSize', f2)
        elseif pval < 0.001
            text(pos, maxY+scale*dpt, sprintf('***'), 'color', color, 'horizontalalignment', 'center', 'FontSize', f2)
        end
    else
        text(pos, maxY-(scale+.5)*dpt, sprintf('%.2g', pval), 'color', color, 'horizontalalignment', 'center', 'FontSize', f2-4)
    end

    plot([pos-width, pos+width], [maxY-(scale)*dpt, maxY-(scale)*dpt], ['-', color], 'LineWidth', 1)
end

end

function sam_style()
% figure style
set(gca, 'FontSize', 12);
set(gca, 'LineWidth', 1.5);
box off;
set(gca, 'TickDir', 'out');
set(gca, 'TickLength', 0.01*[1, 1]);
set(get(gca, 'XLabel'), 'FontSize', 16);
set(get(gca, 'YLabel'), 'FontSize', 16);
set(gcf, 'Color', 'w');
end
