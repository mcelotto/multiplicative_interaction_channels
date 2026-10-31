%%%%%%%%%%%%%%%%%%%%%%%
% simulations shown in Fig. 2h-i and Fig. S2
% data-efficiency of MIC and AIS compared to full-rank models
%%%%%%%%%%%%%%%%%%%%%%%

% simulation settings
dim_list = [10 15 20]; % pop dims nS = nX = nY
Nlist = round(logspace(log10(20), log10(20000), 10)); % num samples per sim
noise = 1; % emission noise
mod_stren = 1; % MIC strength
num_sims = 30; % num sims per condition
mA = 2; mC = 1; % ground truth ranks

% fitting parameters
params = struct;
params.max_lin_rank = 10;
params.K = 10;
params.ctgs_folds_on = false;
params.seed = 42; 
params.opts = struct;
params.max_mod_rank = 3;
params.stratify_by = [];

% store results here
results = cell(length(dim_list), length(Nlist), num_sims);
results_full = cell(length(dim_list), length(Nlist), num_sims);
data = cell(length(dim_list), length(Nlist), num_sims);

% iterate over different pop dimensionalities and sample counts
for lll = 1:length(dim_list)
    nS = dim_list(lll); nY = dim_list(lll); nX = dim_list(lll);
    for iii = 1:length(Nlist)
        N = Nlist(iii);
        for jjj = 1:num_sims
            disp([lll iii jjj]); rng("shuffle");

            % randomly oriented AIS and MIC
            UA = randn(nS, mA); UA = UA./vecnorm(UA, 2, 1);
            VA = randn(nY, mA); VA = VA./vecnorm(VA, 2, 1);
            A = UA*VA';

            WS = randn(nS, mC); WS = mod_stren*WS./vecnorm(WS, 2, 1);
            WY = randn(nY, mC); WY = WY./vecnorm(WY, 2, 1);
            WX = randn(nX, mC); WX = WX./vecnorm(WX, 2, 1);

            % simulate
            S = randn(N, nS);
            X = randn(N, nX);
            Y = S*A + ((S*WS).*(X*WX))*WY.' + noise*randn(N, nY);

            % fit with rank selection
            results{lll, iii, jjj} = MIC.model_wrapper(S, X, Y, params);

            % fit wth full rank
            params2 = params;
            params2.mA = results{lll, iii, jjj}.mA_opt; 
            params2.mB = results{lll, iii, jjj}.mB_opt; 
            params2.mD = results{lll, iii, jjj}.mD_opt;
            params2.mC = "full";
            results_full{lll, iii, jjj} = MIC.model_wrapper(S, X, Y, params2);  

            % store simulated data and ground-truth model
            abc = struct;
            abc.S = S; abc.Y = Y; abc.X = X; abc.WS = WS; abc.WY = WY; abc.WX = WX; abc.A = A;
            data{lll, iii, jjj} = abc;
        end
    end
end

% compute EV of ground-truth model (used in scaling collapse of EV curves)
EV_A_gt = nan(size(data)); % AIS
EV_mod_gt = nan(size(data)); % MIC
for l = 1:length(dim_list)
    for i = 1:length(Nlist)
        for j = 1:num_sims
            cvp = results{l, i, j}.cvB{1}.cv_partition;
            S = data{l, i, j}.S; X = data{l, i, j}.X; Y = data{l, i, j}.Y;
            Agt = data{l, i, j}.A; WSgt = data{l, i, j}.WS; WYgt = data{l, i, j}.WY; WXgt = data{l, i, j}.WX;

            evAgt = []; evCgt = [];
            for k = 1:params.K
                % split data
                tr = training(cvp,k);    
                te = test(cvp,k);
                [Strain, Stest] = MIC.split_normalize(S, tr, te);
                [Xtrain, Xtest] = MIC.split_normalize(X, tr, te);
                [Ytrain, Ytest] = MIC.split_normalize(Y, tr, te);
                
                % predictions
                YpredA = Stest*Agt;
                YpredAC = Stest*Agt + ((Stest*WSgt).*(Xtest*WXgt))*WYgt.';

                % performance
                evAgt(k) = MIC.compute_EV(Ytrain, Ytest, YpredA);
                evCgt(k) = MIC.compute_EV(Ytrain, Ytest, YpredAC) - MIC.compute_EV(Ytrain, Ytest, YpredA);
            end

            EV_A_gt(l, i, j) = mean(evAgt);
            EV_mod_gt(l, i, j) = mean(evCgt);
        end
    end
end

% reorganize results
EVA = nan(length(dim_list), length(Nlist), num_sims);
EVAfull = nan(size(EVA));
EVmod = nan(size(EVA));
EVmodfull = nan(size(EVA));
for l = 1:length(dim_list)
    for i = 1:length(Nlist)
        for j = 1:num_sims
            EVA(l, i, j) = results{l, i, j}.veA(results{l, i, j}.mA_try==results{l, i, j}.mA_opt);
            EVAfull(l, i, j) = results{l, i, j}.veA(end);
            EVmod(l, i, j) = results{l, i, j}.ve_mod(results{l, i, j}.mC_try==results{l, i, j}.mC_opt);
            EVmodfull(l, i, j) = results_full{l, i, j}.ve_mod;
        end
    end
end

figure; 
% EV vs. # samples for different pop dimensionalities
for l = 1:length(dim_list)
    subplot(2, length(dim_list)+2, l); hold on;
    errorbar(Nlist, mean(squeeze(EVA(l, :, :)), 2), std(squeeze(EVA(l, :, :)), [], 2)/sqrt(size(squeeze(EVA(l, :, :)), 2)), 'LineWidth', 1.5);  
    errorbar(Nlist, mean(squeeze(EVAfull(l, :, :)), 2), std(squeeze(EVAfull(l, :, :)), [], 2)/sqrt(size(squeeze(EVA(l, :, :)), 2)), 'LineWidth', 1.5); 
    set(gca, 'xscale', 'log');

    subplot(2, length(dim_list)+2, l+length(dim_list)+2); hold on;
    errorbar(Nlist, mean(squeeze(EVmod(l, :, :)), 2), std(squeeze(EVmod(l, :, :)), [], 2)/sqrt(size(squeeze(EVmod(l, :, :)), 2)), 'LineWidth', 1.5);  
    errorbar(Nlist, mean(squeeze(EVmodfull(l, :, :)), 2), std(squeeze(EVmodfull(l, :, :)), [], 2)/sqrt(size(squeeze(EVmod(l, :, :)), 2)), 'LineWidth', 1.5); 
    set(gca, 'xscale', 'log');
end

% scaling collapse (rescale #samples by #params, EV by EV of ground-truth model)
for l = 1:length(dim_list)
    subplot(2, 5, 10); hold on;
    cc = squeeze(EVmodfull(l, :, :))/mean(squeeze(evCgt_a(l, end, :)), 1);
    errorbar(Nlist/(dim_list(l)^3), mean(cc, 2), std(cc, [], 2)/sqrt(size(cc, 2)), 'LineWidth', 1.5); 

    subplot(2, 5, 9); hold on;
    cc = squeeze(EVmod(l, :, :))/mean(squeeze(evCgt_a(l, end, :)), 1);
    errorbar(Nlist/(dim_list(l)), mean(cc, 2), std(cc, [], 2)/sqrt(size(cc, 2)), 'LineWidth', 1.5);

    subplot(2, 5, 5); hold on;
    cc = squeeze(EVAfull(l, :, :))/mean(squeeze(evAgt_a(l, end, :)), 1);
    errorbar(Nlist/(dim_list(l)^2), mean(cc, 2), std(cc, [], 2)/sqrt(size(cc, 2)), 'LineWidth', 1.5); 

    subplot(2, 5, 4); hold on;
    cc = squeeze(EVA(l, :, :))/mean(squeeze(evAgt_a(l, end, :)), 1);
    errorbar(Nlist/(dim_list(l)), mean(cc, 2), std(cc, [], 2)/sqrt(size(cc, 2)), 'LineWidth', 1.5);
end

set(gca, 'xscale', 'log');