%%%%%%%%%%%%%%%%%%%%%%%%
% code to generate fig S10m
% MIC-AIS alignment of ACA and VIP weights after refitting MICs with ACA
% axons randomly subsampled to half (N/2) or a quarter (N/4) of each
% session's population, with ranks fixed at those selected in fig 4
%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

results_path = @@ your path; % folder with the results of fig4eh.m
save_path = @@ your path; % folder for fitting results
fig_save_path = @@ your path; % folder for figures

% selected ranks and sessions with significant modulation (fig 4)
load(fullfile(results_path, 'ACAxVIP_fig4_results'), "res", "keep");

% fitting settings
params = struct;
params.max_lin_rank = 20;
params.K = 10;
params.ctgs_folds_on = true;
params.seed = 42;
params.opts = struct;
params.stratify_by = [];
params.max_mod_rank = 10;
params.nshuff = 10; % number of random subsamples per session

% store results here
res2 = cell(28, params.nshuff); % N/2 ACA axons
res4 = cell(28, params.nshuff); % N/4 ACA axons

% refit MICs on subsampled ACA populations
for idx = 1:28
    disp(idx)
    if keep(idx)
        [ACA, VIP, wheel, pupil] = readACAVIP(idx); % read data
        disp(size(VIP)); disp(size(ACA))
        nACA = size(ACA, 2);

        for k = 1:params.nshuff
            disp(['resample ', num2str(k)])
            rng(idx*params.nshuff+k)

            params2 = params;
            params2.mA = res{idx}.mA_opt;
            params2.mB = res{idx}.mB_opt;
            params2.mC = res{idx}.mC_opt;
            params2.mD = res{idx}.mD_opt;

            % N/2
            selACA = randperm(nACA);
            selACA = selACA(1:ceil(nACA/2));
            X = [wheel pupil];
            S = ACA(:, selACA);
            Y = VIP;
            res2{idx, k} = MIC.model_wrapper(S, X, Y, params2);

            % N/4
            selACA = randperm(nACA);
            selACA = selACA(1:ceil(nACA/4));
            X = [wheel pupil];
            S = ACA(:, selACA);
            Y = VIP;
            res4{idx, k} = MIC.model_wrapper(S, X, Y, params2);
        end
    end
end

% save
save(fullfile(save_path, 'half_quarter_ACA_resamp'), "res2", "res4", "params", "keep");

%% alignment indices, averaged across subsamples
load(fullfile(save_path, 'half_quarter_ACA_resamp'));

res_sub = {res2, res4};
sub_names = {'N/2 ACA axons', 'N/4 ACA axons'};
num_shufs = 5e3; % number of random rotations

fig = figure;
for r = 1:2
    res = res_sub{r};
    align_hat = nan(28, params.nshuff, 2);
    align_shuf = nan(28, params.nshuff, 2);
    for i = find(keep)
        for k = 1:params.nshuff
            if res{i, k}.mC_opt>0
                Ahat = mean(res{i, k}.cvA{res{i, k}.mA_try==res{i, k}.mA_opt}.A, 3);
                WShat = mean(res{i, k}.cv_mod{res{i, k}.mC_try==res{i, k}.mC_opt}.WS, 3);
                WYhat = mean(res{i, k}.cv_mod{res{i, k}.mC_try==res{i, k}.mC_opt}.WY, 3);
                angles = compute_angles(Ahat, res{i, k}.mA_opt, WShat, WYhat);
                al_idx.source = cos(deg2rad(angles.source)).^2;
                al_idx.target = cos(deg2rad(angles.target)).^2;
                align_hat(i, k, :) = [mean(al_idx.source) mean(al_idx.target)];

                % random rotation null
                temp = zeros(1, 2);
                for j = 1:num_shufs
                    angles = compute_angles(Ahat, res{i, k}.mA_opt, randn(size(WShat)), randn(size(WYhat)));
                    al_idx.source = cos(deg2rad(angles.source)).^2;
                    al_idx.target = cos(deg2rad(angles.target)).^2;
                    temp = temp + [mean(al_idx.source) mean(al_idx.target)]/num_shufs;
                end
                align_shuf(i, k, :) = temp;
            end
        end
    end

    align_hat = align_hat(keep, :, :); align_shuf = align_shuf(keep, :, :);
    align_idx = (align_hat-align_shuf)./(1-align_shuf);
    align_idx = squeeze(mean(align_idx, 2)); % sessions x (ACA, VIP)

    % plot AIS-MIC alignment indices
    subplot(1, 2, r); hold on
    xshift = linspace(-0.1, 0.1, size(align_idx, 1));
    valid = ~isnan(sum(align_idx, 2));
    bar(1, mean(align_idx(:, 1), 'omitnan'), 'FaceColor', 'r')
    errorbar(1, mean(align_idx(:, 1), 'omitnan'), std(align_idx(:, 1), 'omitnan')/sqrt(sum(valid)), 'k', 'LineStyle', 'none')
    scatter(1+xshift, align_idx(:, 1), 'MarkerFaceColor', [0.6, 0.6, 0.6], 'MarkerEdgeColor', [0.6, 0.6, 0.6])

    bar(2, mean(align_idx(:, 2), 'omitnan'), 'FaceColor', 'b')
    errorbar(2, mean(align_idx(:, 2), 'omitnan'), std(align_idx(:, 2), 'omitnan')/sqrt(sum(valid)), 'k', 'LineStyle', 'none')
    scatter(2+xshift, align_idx(:, 2), 'MarkerFaceColor', [0.6, 0.6, 0.6], 'MarkerEdgeColor', [0.6, 0.6, 0.6])
    plot([1, 2]'+xshift, align_idx', '-', 'Color', [0.6, 0.6, 0.6])
    ylabel('Align idx')
    [~, p] = ttest(align_idx(:, 1), align_idx(:, 2), 'Tail', 'Left'); % one-tailed paired t-test
    pvalues_plot(p, 1.5, 0.9, 0.05, 1, 18, 0, 'k', 0)
    xticks([1, 2])
    xticklabels({'ACA', 'VIP'})
    title(sub_names{r})
    sam_style
end
figure_save(fig_save_path, 'align_idx_subsampled_ACA', fig)

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

% angles (deg) between MIC weights and source / target AIS
function angles = compute_angles(A, mA, WS, WY)
[U, ~, V] = svd(A', "econ"); % U output, V input (A is nS x nY)
U = U(:, 1:mA); V = V(:, 1:mA);
PY = U*U'; PS = V*V';
angles = struct;
for i = 1:size(WS, 2)
    tmp = norm(WS(:, i)'*PS, "fro")^2/norm(WS(:,i), "fro")^2; % cos^2(theta_src)
    angles.source(i) = rad2deg(acos(sqrt(tmp)));
    tmp = norm(PY*WY(:, i), "fro")^2/norm(WY(:,i), "fro")^2; % cos^2(theta_targ)
    angles.target(i) = rad2deg(acos(sqrt(tmp)));
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
