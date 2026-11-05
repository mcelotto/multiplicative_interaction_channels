%%%%%%%%%%%%%%%%%%%%%%%%%
% code to generate fig S9e (V1-V2 data) and fig S11g (ACA-VIP data)
% bilinearity diagnostic (Appendix B.11) applied to the fitted MICs of
% sessions with significant modulation
%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

dataset = 'V1_V2'; % 'V1_V2' (fig S9e) or 'ACA_VIP' (fig S11g)
results_path = @@ your path; % folder with the results of fig4ad_and_S9.m (V1-V2) or fig4eh.m (ACA-VIP)
fig_save_path = @@ your path;

% load fitted models (res) and sessions with significant modulation (keep)
switch dataset
    case 'V1_V2'
        load(fullfile(results_path, 'one_hot_sig_heldout_V1_V2'), "res", "keep", "params");
    case 'ACA_VIP'
        load(fullfile(results_path, 'ACAxVIP_fig4_results'), "res", "keep");
end

% diagnostic on each MIC, averaged across the MICs of a session
dX = nan(numel(res), 1); % Delta EV_NL, nonlinearity in x
dZ = nan(numel(res), 1); % Delta EV_NL, nonlinearity in z
for idx = find(keep(:))'
    % read data
    switch dataset
        case 'V1_V2'
            [V1, V2, stim] = readSemedo(idx, params);
            S = V1'; X = stim'; Y = V2';
        case 'ACA_VIP'
            [ACA, VIP, wheel, pupil] = readACAVIP(idx);
            S = ACA; X = [wheel pupil]; Y = VIP;
    end
    valid = sum(isnan([S X Y]), 2)==0; % same samples as in model_wrapper
    S = S(valid, :); X = X(valid, :); Y = Y(valid, :);

    dX_chan = nan(1, res{idx}.mC_opt);
    dZ_chan = nan(1, res{idx}.mC_opt);
    for chan = 1:res{idx}.mC_opt
        disp(['sess ', num2str(idx), ' | MIC ', num2str(chan)])
        [sX, sZ, sY, foldid] = MIC.heldout_scores(res{idx}, S, X, Y, chan);
        out = MIC.bilinearity_test(sX, sZ, sY, foldid);
        dX_chan(chan) = out.X.dve;
        dZ_chan(chan) = out.Z.dve;
    end
    dX(idx) = mean(dX_chan, 'omitnan');
    dZ(idx) = mean(dZ_chan, 'omitnan');
end
dX = dX(keep); dZ = dZ(keep);

% one-tailed sign test across sessions (H0: P(Delta EV_NL > 0) = 0.5)
d = [dX dZ];
names = {'X', 'Z'};
fprintf('\n%s: %d/%d sessions pass the modulation gate\n', dataset, sum(~isnan(dX)), numel(dX));
for t = 1:2
    v = d(~isnan(d(:, t)), t);
    p = 1 - binocdf(sum(v > 0) - 1, numel(v), 0.5);
    fprintf('  %s: %d/%d sessions dve > 0 | sign test p = %.3g\n', names{t}, sum(v > 0), numel(v), p);
end

% plot: sessions (grey), mean +- SEM (red)
fig = figure('Color', 'w', 'Position', [200, 254, 380, 420]); hold on;
xshift = linspace(-0.12, 0.12, size(d, 1))';
for t = 1:2
    scatter(t + xshift, d(:, t), 20, 'filled', 'MarkerFaceColor', [0.5, 0.5, 0.5]);
    scatter(t, mean(d(:, t), 'omitnan'), 50, 'filled', 'MarkerFaceColor', [0.8, 0.1, 0.1]);
    errorbar(t, mean(d(:, t), 'omitnan'), std(d(:, t), 'omitnan')/sqrt(sum(~isnan(d(:, t)))), ...
        'LineStyle', 'none', 'Color', [0.8, 0.1, 0.1], 'LineWidth', 2);
end
yline(0, '--');
xticks(1:2); xticklabels(names); xlim([0.5 2.5]);
ylabel('Heldout \Delta EV (nonlin / total)')
title([strrep(dataset, '_', '-'), ' dataset'], 'FontSize', 13)
sam_style;
figure_save(fig_save_path, [dataset, '_bilin_diag'], fig)

%%%%%%% HELPERS %%%%%%
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
