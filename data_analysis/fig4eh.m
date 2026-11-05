%%%%%%%%%%%%%%%%%%%%%%%%
% code to generate fig 4
% behavioral state modulation of ACA-VIP interactions
%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

save_path = @@ your path; % folder for fitting results

% fitting settings
params = struct;
params.max_lin_rank = 20;
params.K = 10;
params.ctgs_folds_on = true; 
params.seed = 42; 
params.opts = struct;
params.stratify_by = [];
params.max_mod_rank = 10;

% other settings
nrot = 1000; % number of random rotations used to estimate null for geometry indices
Kouter = 2; nshuffle = 2e3; shuffle_type = "circshift"; % shuffle test settings
sig_level = 0.05;

% store results here
res = cell(28, 1); % main pipeline
res_cstr = cell(28, 3); % constrained fits
alpha_null_subtr = nan(28, 4); % null subtracted geometry indices
p_A = nan(28, 1); p_mod = nan(28, 1); % AIS and MIC p-values
res_shuf = cell(28, 1); % complete output of the shuffle test
cc_all = cell(28, 1); % correlation slopes

% run main pipeline, constrained fits, and significance test on all
% sessions
for idx = 1:28
    disp(idx)
    [ACA, VIP, wheel, pupil, stim_on, t] = readACAVIP(idx); % read data
    disp(size(VIP)); disp(size(ACA))
    X = [wheel pupil];
    S = ACA;
    Y = VIP;

    res{idx} = MIC.model_wrapper(S, X, Y, params);

    if res{idx}.mC_opt>0
        % constrained fits (re-use additive term ranks, re-select mod rank but
        % require it to be <= mod rank for unconstrained fit)
        params2 = params;
        params2.max_mod_rank = res{idx}.mC_opt;
        params2.mA = res{idx}.mA_opt; params2.mB = res{idx}.mB_opt; params2.mD = res{idx}.mD_opt;
    
        params2.opts.source_constraint = "same"; params2.opts.target_constraint = "same";
        res_cstr{idx, 1} = MIC.model_wrapper(S, X, Y, params2); % MIC confined to AIS in source and target
    
        params2.opts.source_constraint = "none"; params2.opts.target_constraint = "same";
        res_cstr{idx, 2} = MIC.model_wrapper(S, X, Y, params2); % MIC confined to AIS in target
    
        params2.opts.source_constraint = "same"; params2.opts.target_constraint = "none";
        res_cstr{idx, 3} = MIC.model_wrapper(S, X, Y, params2); % MIC confined to AIS in source
    
        % compute geometry indices
        Ahat = mean(res{idx}.cvA{res{idx}.mA_try==res{idx}.mA_opt}.A, 3);
        WShat = mean(res{idx}.cv_mod{res{idx}.mC_try==res{idx}.mC_opt}.WS, 3);
        WYhat = mean(res{idx}.cv_mod{res{idx}.mC_try==res{idx}.mC_opt}.WY, 3);
    
        abc = MIC.compute_alphas(Ahat, res{idx}.mA_opt, WShat, WYhat);
        alpha_null_subtr(idx, :) = [mean(abc.in) mean(abc.Srot) mean(abc.Yrot) mean(abc.new)]; % averaged across MICs

        % random rotation null
        alpha_null = nan(nrot, 4);
        for i = 1:nrot
            abc = MIC.compute_alphas(Ahat, res{idx}.mA_opt, randn(size(WShat)), randn(size(WYhat)));
            alpha_null(i, :) = [mean(abc.in) mean(abc.Srot) mean(abc.Yrot) mean(abc.new)];
        end
    
        alpha_null_subtr(idx, :) = alpha_null_subtr(idx, :) - mean(alpha_null, 1);
    
        % compute correlation slopes
        WXhat = mean(res{idx}.cv_mod{res{idx}.mC_try==res{idx}.mC_opt}.WX, 3);
    
        WShat = WShat./vecnorm(WShat, 2, 1);
        WYhat = WYhat./vecnorm(WYhat, 2, 1);
        WXhat = WXhat./vecnorm(WXhat, 2, 1);
    
        qq = linspace(0, 1, 7); % quantiles of modulator projection
        cc = zeros(size(WShat, 2), length(qq)-1); % correlation between ACA and VIP projections
        cols = cool(length(qq)-1);
        for j = 1:size(WShat, 2)
            ss = S*WShat(:, j); xx = X*WXhat(:, j); yy = Y*WYhat(:, j); % projections
            for k = 1:(length(qq)-1)
                if k == (length(qq)-1)
                    keep2 = quantile(xx, qq(k))<=xx & xx<=quantile(xx, qq(k+1));
                else
                    keep2 = quantile(xx, qq(k))<=xx & xx<quantile(xx, qq(k+1));
                end
                cc(j, k) = corr(ss(keep2), yy(keep2));
    
                % scatterplot of projections for example session
                if idx==27 & j==1
                    figure(1);
                    subplot(2, 14, [6:9 20:23]); hold on;
                    scatter(ss(keep2), yy(keep2), 10, cols(k, :), 'filled', 'o', 'MarkerFaceAlpha', 0.05, 'MarkerEdgeColor', 'none');
                    p = polyfit(ss(keep2), yy(keep2), 1);
                    plot([-5 5], [-5 5]*p(1)+p(2), 'Color', cols(k, :), 'LineWidth', 2);
                end
            end
        end
        cc_all{idx} = cc;
    end

    % run shuffle test to assess significance of AIS and MIC 
    [inner_model, EV_A, EV_B, EV_D, EV_mod, EV_A_shuf, EV_B_shuf, EV_D_shuf, EV_mod_shuf, p_A(idx), p_B, p_D, p_mod(idx)] = MIC.heldout_test(S, X, Y, params, Kouter, nshuffle, shuffle_type);
    abc = struct; abc.inner_model = inner_model; abc.EV_A = EV_A; abc.EV_mod = EV_mod; abc.EV_A_shuf = EV_A_shuf; abc.EV_mod_shuf = EV_mod_shuf; abc.p_A = p_A; abc.p_mod = p_mod;
    res_shuf{idx} = abc;
end

% identify sessions with significant modulation
mC = nan(1, 28); for i = 1:28; mC(i) = res{i}.mC_opt*(p_mod(i)<0.05); end
keep = mC>0;

% save
save(fullfile(save_path, 'ACAxVIP_fig4_results'), "res", "res_cstr", "alpha_null_subtr", "res_shuf", "keep", "params");

%% load results and plot
load(fullfile(save_path, 'ACAxVIP_fig4_results'));
mC = nan(1, 28); for i = 1:28; mC(i) = res{i}.mC_opt*keep(i); end % selected mod rank of sessions with significant modulation

% reorganize EV (columns in plotted order: none, ACA, VIP, both)
cond_names = {'none', 'ACA', 'VIP', 'both'};
EVmod = nan(28, 4); 
EVA = nan(28, 1);
for i = 1:28
    EVA(i) = res{i}.veA(res{i}.mA_opt==res{i}.mA_try);
    if res{i}.mC_opt>0
        EVmod(i, 1) = ve_at_opt(res{i}); % unconstrained
        EVmod(i, 2) = ve_at_opt(res_cstr{i, 3}); % MIC confined to AIS in source (ACA)
        EVmod(i, 3) = ve_at_opt(res_cstr{i, 2}); % MIC confined to AIS in target (VIP)
        EVmod(i, 4) = ve_at_opt(res_cstr{i, 1}); % MIC confined to AIS in source and target (both)
    end
end

% LME statistics: per-session values modeled as d ~ 1 + (1|animal),
% intercept tested against zero (two-tailed); Holm correction across the 4
% constrained-fit comparisons and across the 4 geometry indices (Appendix B.8, B.14)
animal_id = [repelem(1, 1, 10), repelem(2, 1, 10), repelem(3, 1, 8)]; % sessions 1-10: animal 1; 11-20: animal 2; 21-28: animal 3
animal = categorical(animal_id(keep)');
uA = categories(animal);
nA = numel(uA);

% constrained fits: differences in modulation EV between pairs of fits
pairs = [3 4; 2 3; 2 4; 1 3]; % VIP vs both (ns), ACA vs VIP, ACA vs both (sn), none vs VIP (sn + nn)
M = EVmod(keep, :);
stat_cstr = lme_test(M(:, pairs(:, 2)) - M(:, pairs(:, 1)), animal);
stat_cstr.p_holm = holm_adjust(stat_cstr.p_raw);
stat_cstr.Properties.RowNames = strcat(cond_names(pairs(:, 1))', {' vs '}, cond_names(pairs(:, 2))');
disp('--- constrained fits (EV mod), LME ---'); disp(stat_cstr)

% geometry indices: null-subtracted index vs 0
idx_names = {'ss', 'ns', 'sn', 'nn'};
stat_alpha = lme_test(alpha_null_subtr(keep, :), animal);
stat_alpha.p_holm = holm_adjust(stat_alpha.p_raw);
stat_alpha.Properties.RowNames = idx_names;
disp('--- geometry indices (null subtracted), LME ---'); disp(stat_alpha)

% constrained fits; dots: animal means; red: mean +- SEM across sessions
figure(1);
subplot(2, 14, 25:28); hold on;
M_anim = nan(nA, 4);
for a = 1:nA
    M_anim(a, :) = mean(M(animal == uA{a}, :), 1, 'omitnan');
end
for i = 1:4
    scatter(i + 0.05*randn(nA, 1), M_anim(:, i), 55, 'k', 'filled', 'o', 'MarkerFaceAlpha', 0.85);
end
scatter(1:4, mean(M, 1, 'omitnan'), 80, 'r', 'filled', 'o');
errorbar(1:4, mean(M, 1, 'omitnan'), std(M, [], 1, 'omitnan')./sqrt(sum(~isnan(M), 1)), ...
    'k', 'LineStyle', 'none', 'LineWidth', 1.5);
yTop = max(M(:), [], 'omitnan');
for k = 1:size(pairs, 1)
    pvalues_plot(stat_cstr.p_holm(k), mean(pairs(k, :)), yTop + k*(yTop/8), yTop/10, 1, 15, diff(pairs(k, :))/2, 'k', 1);
end
xticks(1:4)
xticklabels(cond_names)
ylabel('mod EV')

% modulation EV relative to AIS EV vs. rank
abc = [];
for i = 1:28
    ve = res{i}.ve_mod(:)'/EVA(i);
    abc(i, :) = [ve ve(end)*ones(1, params.max_mod_rank+1-length(ve))];
end
col_insig = [0 0.4470 0.7410]; % sessions without significant modulation (blue)

subplot(2, 14, [1:4 15:18]); hold on; 
plot(0:10, abc(keep, :), 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
plot(0:10, abc(~keep, :), 'Color', col_insig, 'LineWidth', 1.5);
plot(0:10, nanmean(abc(keep, :), 1), 'Color', 'r', 'LineWidth', 2.5);

for i = 1:28
    if keep(i)
        scatter(mC(i), abc(i, mC(i)+1), 60, 'k', 'filled');
    end
end
ylabel('EV mod / EV AIS')
xlabel('modulation rank')

% geometry indices; dots: animal means; red: mean +- SEM across animals
subplot(2, 14, 11:14); hold on;
alpha_keep = alpha_null_subtr(keep, :);
alpha_anim = nan(nA, 4);
for a = 1:nA
    alpha_anim(a, :) = mean(alpha_keep(animal == uA{a}, :), 1, 'omitnan');
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
            text(pos,maxY+scale*dpt,sprintf('%.1g', pval),'color',color,'horizontalalignment','center','FontSize',f2-4)
        elseif (pval <= 0.055 && pval > 0.01)
            text(pos,maxY+scale*dpt,sprintf('*'),'color',color,'horizontalalignment','center','FontSize',f2)
        elseif (pval <= 0.01 && pval > 0.001)
            text(pos,maxY+scale*dpt,sprintf('**'),'color',color,'horizontalalignment','center','FontSize',f2)
        elseif pval <= 0.001
            text(pos,maxY+scale*dpt,sprintf('***'),'color',color,'horizontalalignment','center','FontSize',f2)
        end
    else
        text(pos,maxY+scale*dpt,sprintf('%.2g', pval),'color',color,'horizontalalignment','center','FontSize',f2-4)
    end

    plot([pos-width,pos+width],[maxY+(scale-.5)*dpt,maxY+(scale-.5)*dpt],['-',color],'LineWidth',1)

elseif maxY < 0
    dpt = -dpt;
    if star
        if pval > 0.055
            text(pos,maxY+scale*dpt,sprintf('%.1g', pval),'color',color,'horizontalalignment','center','FontSize',f2-4)
        elseif (pval < 0.055 && pval > 0.01)
            text(pos,maxY+scale*dpt,sprintf('*'),'color',color,'horizontalalignment','center','FontSize',f2)
        elseif (pval < 0.01 && pval > 0.001)
            text(pos,maxY+scale*dpt,sprintf('**'),'color',color,'horizontalalignment','center','FontSize',f2)
        elseif pval < 0.001
            text(pos,maxY+scale*dpt,sprintf('***'),'color',color,'horizontalalignment','center','FontSize',f2)
        end
    else
        text(pos,maxY-(scale+.5)*dpt,sprintf('%.2g', pval),'color',color,'horizontalalignment','center','FontSize',f2-4)
    end

    plot([pos-width,pos+width],[maxY-(scale)*dpt,maxY-(scale)*dpt],['-',color],'LineWidth',1)
end

end