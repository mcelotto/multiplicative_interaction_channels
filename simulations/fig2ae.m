%%%%%%%%%%%%%%%%%%%%%%%
% simulations shown in Fig. 2a-e
% 2D X, 1D Y, 1D Z, first dim of X is transmissive, second is modulated by Z
%%%%%%%%%%%%%%%%%%%%%%%

num_repeats = 5; % number of simulations per condition
gamma_list = 0:0.2:1.2; % strength of MIC
N = 1e4; % number of samples per simulation
noiseY = 1; % emission noise standard deviation

% fitting parameters
params = struct;
params.seed = 42;
params.K = 10;
params.ctgs_folds_on = false;
params.opts = struct;
params.max_mod_rank = 2;
params.max_lin_rank = 1;
params.stratify_by = [];

results = cell(length(gamma_list), num_repeats); % store fitting results here
EV_A = nan(length(gamma_list), num_repeats); % AIS explained variance
EV_A1 = nan(size(EV_A)); % explained variance of AIS restricted to dim 1 of X
EV_A2 = nan(size(EV_A)); % explained variance of AIS restricted to dim 2 of X
EV_mod = nan(size(EV_A)); % modulatory explained variance
mA = nan(size(EV_A)); % inferred AIS rank
mC = nan(size(EV_A)); % inferred MIC rank

for i = 1:length(gamma_list)
    for j = 1:num_repeats
        rng("shuffle")
        disp([i, j]);
        A = [1 ; 0];
        WX = gamma_list(i)*[0 ; 1]; WY = 1; WZ = 1; % ground-truth MIC
        
        % simulate
        Z = randn(N, 1);
        X = randn(N, 2);
        Y = X*A + ((X*WX).*(Z*WZ))*WY.' + noiseY*randn(N, 1);
    
        % fit
        results{i, j} = MIC.model_wrapper(X, Z, Y, params);

        % store results
        mA(i, j) = results{i, j}.mA_opt;
        mC(i, j) = results{i, j}.mC_opt;

        EV_A(i, j) = results{i, j}.veA(results{i, j}.mA_try==mA(i, j));
        EV_mod(i, j) = results{i, j}.ve_mod(results{i, j}.mC_try==mC(i, j));

        % re-fit AIS constrained to dim 1 and dim 2 of X
        params2 = params;
        params2.mA = mA(i, j);
        params2.mC = 0;
        params2.mB = 0;
        params2.mD = 0;
        res1 = MIC.model_wrapper(X(:, 1), Z, Y, params2);
        res2 = MIC.model_wrapper(X(:, 2), Z, Y, params2);
        
        EV_A1(i, j) = res1.veA(1);
        EV_A2(i, j) = res2.veA(1);
    end

end

% plot
figure;

subplot(1, 4, 3); hold on;
plot(gamma_list, mean(mA, 2), 'sb', 'MarkerFaceColor', 'b', 'MarkerSize', 16);
plot(gamma_list, mean(mC, 2), 'og', 'MarkerFaceColor', 'g', 'MarkerSize', 12);

subplot(1, 4, 4); hold on;
plot(gamma_list, mean(EV_A, 2), 'sb', 'MarkerFaceColor', 'b', 'MarkerSize', 16);
plot(gamma_list, mean(EV_A1, 2), 'k', 'LineWidth', 1.5);
plot(gamma_list, mean(EV_A2, 2), 'k', 'LineWidth', 1.5);
plot(gamma_list, mean(EV_mod, 2), 'og', 'MarkerFaceColor', 'g', 'MarkerSize', 12);

subplot(1, 4, 1);
scatter(X(:, 1), Y, 30, Z, 'filled', 'o', 'MarkerFaceAlpha', 0.1);

subplot(1, 4, 2);
scatter(X(:, 2), Y, 30, Z, 'filled', 'o', 'MarkerFaceAlpha', 0.1);