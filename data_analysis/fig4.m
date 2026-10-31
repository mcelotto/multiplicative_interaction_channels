%%%%%%%%%%%%%%%%%%%%%%%%
% code to generate fig 4
% behavioral state modulation of ACA-VIP interactions
%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

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
    
        abc = MIC.compute_alphas(Ahat, results.mA_opt, WShat, WYhat);
        alpha_null_subtr(idx, :) = [abc.in abc.Srot abc.Yrot abc.new]; 

        % random rotation null
        alpha_null = nan(nrot, 4);
        for i = 1:nrot
            abc = MIC.compute_alphas(Ahat, res{idx}.mA_opt, randn(size(WShat)), randn(size(WYhat)));
            alpha_null(i, :) = [abc.in abc.Srot abc.Yrot abc.new];
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

% save
save(@@ your path, "res"); save(@@ your path, "res_cstr"); save(@@ your path, "alpha_null_subtr"); save(@@ your path, "res_shuf");

% plot results

% identify sessions with significant modulation
mC = nan(1, 28); for i = 1:28; mC(i) = res{i}.mC_opt*(p_mod(i)<0.05); end
keep = mC>0;
save(@@ save path for keep, "keep");

% reorganize EV
EVmod = nan(28, 4); 
EVA = nan(28, 1);
for i = 1:28
    EVA(i) = res{i}.veA(res{i}.mA_opt==res{i}.mA_try);
    for j = 1:4
        if ~isempty(res_cstr{i, j}.mC_opt)
            EVmod(i, j) = res_cstr{i, j}.ve_mod(res_cstr{i, j}.mC_opt==res_cstr{i, j}.mC_try); 
        end
    end
end

figure(1);
subplot(2, 14, 25:28); hold on;
for i = 1:4
    scatter(i+0.08*randn(sum(keep), 1), EVmod(keep, i), 40, 'k', 'filled', 'o', 'MarkerFaceAlpha', 0.3);
end

scatter(1:4, nanmean(EVmod(keep, :), 1), 80, 'r', 'filled', 'o');
errorbar(1:4, nanmean(EVmod(keep, :), 1), nanstd(EVmod(keep, :), [], 1)./sqrt(sum(keep)), 'k', 'LineWidth', 1.5, 'LineStyle', 'none');

% test significance of constrained EV comparisons
M = EVmod(keep, 1:4);
T = array2table(M, 'VariableNames', {'Cond1','Cond2','Cond3','Cond4'});
WithinDesign = table(categorical([1;2;3;4]), 'VariableNames', {'Condition'}); % define the within-subject factor
rm = fitrm(T, 'Cond1-Cond4 ~ 1', 'WithinDesign', WithinDesign); % fit repeated-measures model
ranovatbl = ranova(rm); % repeated-measures ANOVA for the within-subject effect
posthocTbl = multcompare(rm, 'Condition', 'ComparisonType', 'bonferroni'); % pairwise post hoc comparisons across conditions with Bonferroni correction
posthocTbl = posthocTbl(...
    double(posthocTbl.Condition_2)>double(posthocTbl.Condition_1),:);

for i = 1:size(posthocTbl,1)
    maxY = max(M,[],'all');
    p = posthocTbl.pValue(i);
    pos = mean(double([posthocTbl.Condition_1(i),posthocTbl.Condition_2(i)]));
    diff = double(posthocTbl.Condition_2(i))-double(posthocTbl.Condition_1(i));
    pvalues_plot(p,pos,maxY+i*(maxY/8),(maxY/10),1,15,diff/2,'k',1);
end

% modulation EV vs. rank
abc = []; for i = 1:28; abc(i, :) = [res{i}.ve_mod(:)' res{i}.ve_mod(end)*ones(1, params.max_mod_rank+1-length(res{i}.ve_mod))]; end
col_insig = rand(1, 3);

subplot(2, 14, [1:4 15:18]); hold on; 
plot(0:10, abc(keep, :), 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
plot(0:10, abc(~keep, :), 'Color', col_insig, 'LineWidth', 1.5);
plot(0:10, nanmean(abc(keep, :), 1), 'Color', 'r', 'LineWidth', 2.5);

for i = 1:28
    if keep(i)
        scatter(mC(i), abc(i, mC(i)+1), 60, 'k', 'filled');
    end
end

% plot geometry indices
subplot(2, 14, 11:14); hold on;
for i = 1:4
    scatter(i+0.05*randn(sum(keep), 1), alpha_null_subtr(keep, i), 40, 'filled', 'o', 'MarkerFaceColor', 'k', 'MarkerFaceAlpha', 0.3);
end
scatter(1:4, mean(alpha_null_subtr(keep, :), 1), 80, 'r', 'filled', 'o');
yline(0, '--r', 'LineWidth', 1.5);
errorbar(1:4, mean(alpha_null_subtr(keep, :), 1), std(alpha_null_subtr(keep, :), 1)/sqrt(sum(keep)), 'k', 'LineStyle', 'none', 'LineWidth', 1);

for i = 1:4
    [h, p] = ttest(alpha_null_subtr(keep, i));
    p = p*4; % Bonferroni correction
    pvalues_plot(p,i,0.8,0.05,1,14, 0, 'k', 1)
end

%%%%%%% HELPERS %%%%%%
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