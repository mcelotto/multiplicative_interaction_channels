%%%%%%%%%%%%%%%%%%%%%%%
% simulations shown in Fig. S8
% power of the bilinearity diagnostic (Appendix B.11)
%
% Each nonlinear generator of Appendix B.10 is linearly interpolated with
% a bilinear generator built from the same draw of S, X, A and channel
% axes
%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

results_path = @@ your path;
fig_save_path = @@ your path;

% simulation settings
scenario_list = {'relu', 'tanh', 'exp', 'rot_perp', 'rot_par', 'relu_z'}; % fig S8a-f
noise_list = [1 1 0.3 1 1 1]; % emission noise s.d. per scenario (exp: uniform inputs carry less modulation variance)
rho_list = 0:0.1:1; % interpolation between bilinear and nonlinear generator
a_list = [1/3, 2/3, 1]; % modulation strength
N = 2e4; % number of samples per sim
num_runs = 50; % number of sims per condition
nS = 5; nY = 5; % population dimensionalities (nX = 1; rotations: nS = nY = 3, set in the generator)
mA_gt = 2; % ground-truth AIS rank
alpha = 0.05; % significance level of the sign test

% fitting parameters (ranks fixed at the ground truth, no rank search)
params = struct;
params.max_lin_rank = mA_gt;
params.K = 10;
params.ctgs_folds_on = true;
params.seed = 42;
params.opts = struct;
params.stratify_by = [];
params.max_mod_rank = 1;
params.mA = mA_gt; params.mB = 0; params.mC = 1; params.mD = 0;

% store results
nP = length(scenario_list); nA = length(a_list); nR = length(rho_list);
dveX = nan(nP, nA, nR, num_runs); % Delta EV_NL, nonlinearity in x
dveZ = nan(nP, nA, nR, num_runs); % Delta EV_NL, nonlinearity in z
ve_mod = nan(nP, nA, nR, num_runs); % modulation EV

for p = 1:nP
    params2 = params;
    if strcmp(scenario_list{p}, 'exp')
        params2.mB = 1; % the exponent shifts the mean of Y with X
    elseif strcmp(scenario_list{p}, 'rot_perp')
        params2.mC = 2; % the linearized rotation has rank 2
    end

    for ia = 1:nA
        for ir = 1:nR
            for r = 1:num_runs
                fprintf('%s a=%.2f rho=%.2f run %d\n', scenario_list{p}, a_list(ia), rho_list(ir), r);
                rng("shuffle");

                % simulate
                [S, X, Ydet] = generate_scenario(scenario_list{p}, a_list(ia), rho_list(ir), N, nS, nY, mA_gt);
                Y = Ydet + noise_list(p)*randn(size(Ydet));

                % fit and test the first MIC
                results = MIC.model_wrapper(S, X, Y, params2);
                ve_mod(p, ia, ir, r) = results.ve_mod(1);

                [sX, sZ, sY, foldid] = MIC.heldout_scores(results, S, X, Y, 1);
                out = MIC.bilinearity_test(sX, sZ, sY, foldid);
                dveX(p, ia, ir, r) = out.X.dve;
                dveZ(p, ia, ir, r) = out.Z.dve;
            end
        end
    end
end
save(fullfile(results_path, 'figS8_results'), 'dveX', 'dveZ', 've_mod', 'params', 'scenario_list', 'noise_list', 'rho_list', 'a_list', 'N');

% power: fraction q of runs with Delta EV_NL > 0 (runs failing the
% modulation gate count as <= 0); the one-tailed sign test across n =
% num_runs units rejects if >= kstar units have Delta EV_NL > 0, so that
% power = P(Binom(num_runs, q) >= kstar)
fracX = sum(dveX > 0, 4) / num_runs;
fracZ = sum(dveZ > 0, 4) / num_runs;
kstar = find(1 - binocdf((0:num_runs) - 1, num_runs, 0.5) < alpha, 1) - 1;
powX = 1 - binocdf(kstar - 1, num_runs, fracX);
powZ = 1 - binocdf(kstar - 1, num_runs, fracZ);
evm = mean(ve_mod, 4, 'omitnan');

for p = 1:nP
    for ia = 1:nA
        for ir = 1:nR
            fprintf('%-8s a=%.2f rho=%.2f | EV_mod=%.4f | frac dve>0: X=%.2f Z=%.2f | power: X=%.2f Z=%.2f\n', ...
                scenario_list{p}, a_list(ia), rho_list(ir), evm(p, ia, ir), fracX(p, ia, ir), fracZ(p, ia, ir), ...
                powX(p, ia, ir), powZ(p, ia, ir));
        end
    end
end

% plot: columns = scenario, rows = nonlinearity in x, in z; one line per a
cmap = cool(256);
cols = cmap(round(linspace(60, 256, nA)), :);
row_names = {'nonlin in X', 'nonlin in Z'};
fig = figure('Color', 'w', 'Position', [40 40 270*nP 500]);
for p = 1:nP
    for row = 1:2
        subplot(2, nP, (row-1)*nP + p); hold on;
        for ia = 1:nA
            nm = sprintf('a = %.2g (EV_{mod} = %.1f%%)', a_list(ia), 100*evm(p, ia, 1)); % EV_mod at rho = 0
            if row == 1
                yy = squeeze(powX(p, ia, :));
            else
                yy = squeeze(powZ(p, ia, :));
            end
            plot(rho_list, yy, '-o', 'Color', cols(ia, :), 'LineWidth', 1.6, 'MarkerFaceColor', cols(ia, :), 'MarkerSize', 4, 'DisplayName', nm);
        end
        yline(0.8, ':k', 'HandleVisibility', 'off');
        ylim([0 1.02]); xlim([min(rho_list) max(rho_list)]);
        set(gca, 'FontSize', 10, 'LineWidth', 1.1, 'TickDir', 'out', 'Box', 'off');
        if row == 1
            title(strrep(scenario_list{p}, '_', '\_'), 'FontSize', 12);
            legend('Location', 'southeast', 'FontSize', 6);
        else
            xlabel('rho');
        end
        if p == 1
            ylabel([row_names{row}, ' power'], 'FontSize', 10);
        end
    end
end
sgtitle(['Bilinearity diagnostic (n = ', num2str(num_runs), ' runs)'], 'FontSize', 14);
figure_save(fig_save_path, 'figS8', fig)

%%%%%%% HELPERS %%%%%%
function [S, X, Ydet] = generate_scenario(scenario, a, rho, N, nS, nY, mA_gt)
% noise-free output Ydet = (1 - rho)*Ybil + rho*Ynl, with Ybil and Ynl
% built from the same draw of the ingredients
switch scenario

    case {'relu', 'tanh', 'relu_z'}
        UA = randn(nS, mA_gt);
        VA = randn(nY, mA_gt);
        UA = UA./vecnorm(UA, 2, 1);
        VA = VA./vecnorm(VA, 2, 1);
        A_gt = UA*VA';
        WS_gt = randvec_angle_to_subspace(UA, 0);
        WY_gt = randvec_angle_to_subspace(VA, pi/4);
        WX_gt = 1;
        WS_gt = WS_gt./vecnorm(WS_gt, 2, 1);
        WY_gt = WY_gt./vecnorm(WY_gt, 2, 1);

        S = randn(N, nS);
        X = randn(N, 1);
        Ybil = S*A_gt + a*((S*WS_gt).*(X*WX_gt))*WY_gt.';
        if strcmp(scenario, 'relu')
            Ynl = my_relu(Ybil);
        elseif strcmp(scenario, 'tanh')
            Ynl = tanh(Ybil);
        else
            Ynl = S*A_gt + a*((S*WS_gt).*my_relu(X*WX_gt))*WY_gt.';
        end

    case 'exp'
        % positive A; S and X uniform in [0, 1]
        UA = abs(randn(nS, mA_gt));
        VA = abs(randn(nY, mA_gt));
        UA = UA./vecnorm(UA, 2, 1);
        VA = VA./vecnorm(VA, 2, 1);
        A_gt = UA*VA';
        WS_gt = randvec_angle_to_subspace(UA, 0);
        WY_gt = randvec_angle_to_subspace(VA, pi/4);
        WS_gt = WS_gt./vecnorm(WS_gt, 2, 1);
        WY_gt = WY_gt./vecnorm(WY_gt, 2, 1);

        S = rand(N, nS);
        X = rand(N, 1);
        Ynl = zeros(N, nY);
        for i = 1:N
            base = S(i, :)*A_gt;
            p = 1 + a*(2*X(i) - 1);
            Ynl(i, :) = base.^p;
        end
        Ybil = S*A_gt + a*((S*WS_gt).*(2*X - 1))*WY_gt.'; % modulator: the centred exponent 2X - 1

    case {'rot_perp', 'rot_par'}
        % nS = nY = 3, A = diag(1, 1, 0), rotation amplitude a*pi
        nS = 3; nY = 3;
        if strcmp(scenario, 'rot_par')
            rot_axis = [randn(2, 1) ; 0];
            rot_axis = rot_axis/vecnorm(rot_axis, 2);
        else
            rot_axis = [0 ; 0 ; 1];
        end
        A = [1 0 0 ; 0 1 0 ; 0 0 0];
        WX = randn(1, 1);
        WX = WX./vecnorm(WX, 2, 1);
        rot_amp = a*pi;

        S = randn(N, nS);
        X = randn(N, 1);
        Ynl = zeros(N, nY);
        for i = 1:N
            theta = X(i, :)*WX*rot_amp/3;
            Acurr = rotation_matrix(theta, rot_axis)'*A;
            Ynl(i, :) = S(i, :)*Acurr;
        end

        % bilinear generator on the same ingredients: A = UA*VA', UA = VA = [e1 e2]
        UA = A(:, 1:2);
        VA = A(:, 1:2);
        WS_gt = randvec_angle_to_subspace(UA, 0);
        WY_gt = randvec_angle_to_subspace(VA, pi/4);
        WS_gt = WS_gt./vecnorm(WS_gt, 2, 1);
        WY_gt = WY_gt./vecnorm(WY_gt, 2, 1);
        Ybil = S*A + a*((S*WS_gt).*(X*WX))*WY_gt.';

    otherwise
        error('unknown scenario %s', scenario);
end
Ydet = (1 - rho)*Ybil + rho*Ynl;
end

function x = my_relu(x)
x(x<0) = 0;
end

% draw a unit vector at angle theta from the column space of UA
function x = randvec_angle_to_subspace(UA, theta)
[n, ~] = size(UA);
Q = orth(UA);
a = randn(size(Q, 2), 1);
u = Q*a;
u = u / norm(u);
b = randn(n, 1);
v = b - Q*(Q'*b);
v = v / norm(v);
x = cos(theta)*u + sin(theta)*v;
end

% 3D rotation matrix from axis-angle representation (acts on column vectors)
function R = rotation_matrix(theta, axis)
axis = axis(:);
axis = axis / norm(axis);
ux = axis(1); uy = axis(2); uz = axis(3);
c = cos(theta); s = sin(theta); C = 1 - c;
R = [
    c + ux^2*C,      ux*uy*C - uz*s,  ux*uz*C + uy*s;
    uy*ux*C + uz*s,  c + uy^2*C,      uy*uz*C - ux*s;
    uz*ux*C - uy*s,  uz*uy*C + ux*s,  c + uz^2*C
    ];
end
