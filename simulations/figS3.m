%%%%%%%%%%%%%%%%%%%%%%%
% simulations shown in Fig. S3
% non-Gaussian emission noise
%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc; 
rng("shuffle")

results_path = @@your_results_path;
if ~exist(results_path,'dir')
    mkdir(results_path);
end

% simulation settings
sigmaS = 1; sigmaX = 1; sigmaY = 1;
nS = 5; nX = 5; nY = 5;

params.mod_stren = 1;
params.Nlist = 10000;
params.gamma = 0:0.25:3;

params.num_runs = 50;
nList = 0:5;
mList = 0:5;
lList = 0:5;
params.gt_fit = 1; % if 1, fit model with ground truth ranks

params.seed = 42; params.K = 10; params.ctgs_folds_on = true; params.grid_on = false;
params.max_mod_rank = nS; % note: the max mod_rank may be smaller than 5 sometimes...
params.max_lin_rank = nS; params.stratify_by = [];
params.opts = []; prams.h_base = 2;

% emission noise distributions
models = {'gauss','poiss','zip'};

%%
rng(params.seed);


for obs_model = models

    obs_model = obs_model{1};

    dims_hat.(obs_model) = nan(length(params.Nlist), length(params.gamma), params.num_runs, 3);
    dims_actual.(obs_model) = randi([1], size(dims_hat.(obs_model)));
    mean_princip_angle.(obs_model) = nan(size(dims_hat.(obs_model)));
    EVmod.(obs_model) = nan(length(params.Nlist), length(params.gamma) ,params.num_runs); 
    EVA.(obs_model) = EVmod.(obs_model);
    
    count = 0;
    for NN = 1:length(params.Nlist)
        for g = 1:length(params.gamma)
            for ri = 1:params.num_runs
                count = count+1;
                disp([NN, g, ri])
                rng("shuffle");
                n = dims_actual.(obs_model)(NN, g, ri, 1); 
                m = dims_actual.(obs_model)(NN, g, ri, 2);
                l = dims_actual.(obs_model)(NN, g, ri, 3);
                
                % S is the modulator, X the source, and Y the target
                sim = my_sim_neuro_count_modulation( ...
                    sigmaS, sigmaX, sigmaY, nS, nX, nY, ...
                    0, n, m, l, params.gamma(g), params.Nlist(NN), prams.h_base, prams.h_base, obs_model, count, 0.5, 4);

                S = sim.S';
                X = sim.Xobs';
                Y = sim.Yobs';

                USX = sim.USX; VSX = sim.VSX;
                USY = sim.USY; VSY = sim.VSY;
                UXY = sim.UXY; VXY = sim.VXY;

                WS = sim.WS; WX = sim.WX; WY = sim.WY;

                if params.gt_fit
                    r = dims_actual.(obs_model)(NN, g, ri, 1);
                    c = dims_actual.(obs_model)(NN, g, ri, 2);
                    v = dims_actual.(obs_model)(NN, g, ri, 3);
                    params.mA = r; params.mB = c; params.mC = v; params.mD = 0;
                    tmp = MIC.model_wrapper(S,X,Y,params);
    
                    res{NN,g,ri}.dims = [r, c, v]; % mA, mB, mC
    
                    % Evaluate comm subspaces / mod channels for GT dimensionality
                    tmpA = squeeze(mean(tmp.cvA.A,3));
                    [res{NN,g,ri}.UAhat, res{NN,g,ri}.VAhat] = rank_factorization(tmpA, n);
                    tmpB = squeeze(mean(tmp.cvA.cvBD.B,3));
                    [res{NN,g,ri}.UBhat, res{NN,g,ri}.VBhat] = rank_factorization(tmpB, m);
        
                    % mod channels naming convention follows the old S: source, Y: target, X: modulator
                    res{NN,g,ri}.WX = mean(tmp.cv_mod.WS,3); 
                    res{NN,g,ri}.WY = mean(tmp.cv_mod.WY,3); 
                    res{NN,g,ri}.WZ = mean(tmp.cv_mod.WX,3);

                    EVmod.(obs_model)(NN, g, ri) = tmp.ve_mod;
                    EVA.(obs_model)(NN, g, ri) = tmp.veA;
                else
                    tmp = MIC.model_wrapper(S,X,Y,params);
    
                    res{NN,g,ri}.dims = [tmp.mA_opt, tmp.mB_opt, tmp.mC_opt]; % mA, mB, mC
    
                    % Evaluate comm subspaces / mod channels for GT dimensionality
                    tmpA = squeeze(mean(tmp.cvA{m+1}.A,3));
                    [res{NN,g,ri}.UAhat, res{NN,g,ri}.VAhat] = rank_factorization(tmpA, n);
                    tmpB = squeeze(mean(tmp.cvB{n+1}.B,3));
                    [res{NN,g,ri}.UBhat, res{NN,g,ri}.VBhat] = rank_factorization(tmpB, m);
        
                    % mod channels naming convention follows the old S: source, Y: target, X: modulator
                    res{NN,g,ri}.WX = mean(tmp.cv_mod{l+1}.WS,3); 
                    res{NN,g,ri}.WY = mean(tmp.cv_mod{l+1}.WY,3); 
                    res{NN,g,ri}.WZ = mean(tmp.cv_mod{l+1}.WX,3); 

                    EVmod.(obs_model)(NN, g, ri) = tmp.ve_mod(tmp.mC_opt+1);
                    EVA.(obs_model)(NN, g, ri) = tmp.veA(tmp.mA_opt+1);
                end
               

                dims_hat.(obs_model)(NN,g,ri,:) = res{NN,g,ri}.dims;
                thetaUA = mean(principal_angles(res{NN,g,ri}.UAhat, UXY));
                thetaVA = mean(principal_angles(res{NN,g,ri}.VAhat, VXY));
                thetaUB = mean(principal_angles(res{NN,g,ri}.UBhat, USY));
                thetaVB = mean(principal_angles(res{NN,g,ri}.VBhat, VSY));
                thetaX = mean(principal_angles(res{NN,g,ri}.WX, WS));
                thetaY = mean(principal_angles(res{NN,g,ri}.WZ, WX));
                thetaZ = mean(principal_angles(res{NN,g,ri}.WY, WY));
                mean_princip_angle.(obs_model)(NN, g, ri, :) = [mean([thetaUA, thetaVA]) mean([thetaUB, thetaVB]) mean([thetaX, thetaY, thetaZ])];
                
            end
        end
    end
end

save([results_path,'poiss_gaussLatent_N104_GTranks_',date],'params','res','dims_actual','dims_hat','mean_princip_angle','EVmod','EVA')

%% Plots
save_version = '27-Apr-2026';
load([results_path,'poiss_gaussLatent_N104_GTranks_27-Apr-2026'])

paths.figures = [@@your fig path, save_version];
if ~exist(paths.figures,'dir')
    mkdir(paths.figures)
end

models_labels = {'Gauss','Poisson','Zero-inflated poisson'};

cols = [0.2,0.2,0.8;0.8,0.2,0.8];
clear h
fig = figure('Position',[528,383,1026,231]);
for i = 1:numel(models)
    subplot(1,numel(models),i)
    hold on
    data = EVmod.(models{i});
    h(1) = scatter(params.gamma,squeeze(mean(data,3)),20,'MarkerFaceColor',cols(1,:),'MarkerEdgeColor',cols(1,:));
    errorbar(params.gamma,squeeze(mean(data,3)),squeeze(std(data,[],3)./sqrt(size(data,3))),'LineStyle','none','color',cols(1,:));
    xlabel('gamma')
    ylabel('Mod EV')

    yyaxis right
    data = rad2deg(mean_princip_angle.(models{i})(:,:,:,3));
    h(2) =scatter(params.gamma,squeeze(mean(data,3)),20,'MarkerFaceColor',cols(2,:),'MarkerEdgeColor',cols(2,:));
    errorbar(params.gamma,squeeze(mean(data,3)),squeeze(std(data,[],3)./sqrt(size(EVA.(models{i}),3))),'LineStyle','none','color',cols(2,:));
    ylabel('angle [rad]')
    ylim([0,70])
    xlim([-0.1,params.gamma(end)+0.1])
    ax = gca;
    ax.YAxis(1).Color = cols(1,:);
    ax.YAxis(2).Color = cols(2,:);

    title(models_labels{i})
    if i == 1
        legend(h,{'Mod EV','Angle error'},'Location','NorthEast')
    end
end
figure_save(paths.figures,['EV_angle_nonGauss'],fig)

fig = figure('Position',[528,383,1026,231]);
for i = 1:numel(models)
    subplot(1,numel(models),i)
    hold on
    data = dims_hat.(models{i})==dims_actual.(models{i});
    data = data(:,:,:,3);
    h(1) = scatter(params.gamma,squeeze(mean(data,3)),20,'MarkerFaceColor',cols(1,:),'MarkerEdgeColor',cols(1,:));
    errorbar(params.gamma,squeeze(mean(data,3)),squeeze(std(data,[],3)./sqrt(size(data,3))),'LineStyle','none','color',cols(1,:));
    xlabel('gamma')
    ylabel('m_C accuracy')
    ylim([0,1])

    title(models_labels{i})

end
figure_save(paths.figures,['mCrank_nonGauss'],fig)


% Example distr
clear S X Y h
cnt = 0;
fig=figure('Position',[525,643,1014,241]);
for obs_model = models
    obs_model = obs_model{1}; cnt = cnt+1;
    sim = my_sim_neuro_count_modulation( ...
        sigmaS, sigmaX, sigmaY, nS, nX, nY, ...
        0, n, m, l, params.gamma(g), params.Nlist(NN), prams.h_base, prams.h_base, obs_model, 42, 0.5, 4);
    S.(obs_model) = sim.S';
    X.(obs_model) = sim.Xobs';
    Y.(obs_model) = sim.Yobs';
    subplot(1,numel(models),cnt)
    hold on
    [~,edges] = histcounts([X.(obs_model)(:),Y.(obs_model)(:)],30);
    h(1) = histogram(X.(obs_model),edges,'Normalization','probability');
    h(2) =histogram(Y.(obs_model),edges,'Normalization','probability');
    xlabel('Activity (a.u.)')
    ylabel('prob')
    title(models_labels{cnt})
end
legend(h,{'X','Y'},'Location','NorthEast')
figure_save(paths.figures,['modelsDistr_examples'],fig)

%%

function [S, X, Y, USX, VSX, USY, VSY, UXY, VXY, WS, WX, WY] = ...
    my_sim_poiss_relu(sigmaS, sigmaX, sigmaY, nS, nX, nY, ...
    mSY, mSX, mXY, mSXY, mod_stren, N, baseline_h, poiss_on,seed)

rng(seed)
USX = randn(nX, mSX); VSX = randn(nS, mSX);
USY = randn(nY, mSY); VSY = randn(nS, mSY);
UXY = randn(nY, mXY); VXY = randn(nX, mXY);

USX = USX./vecnorm(USX); VSX = VSX./vecnorm(VSX);
USY = USY./vecnorm(USY); VSY = VSY./vecnorm(VSY);
UXY = UXY./vecnorm(UXY); VXY = VXY./vecnorm(VXY);

WS = randn(nS, mSXY); WX = randn(nX, mSXY); WY = randn(nY, mSXY);
WS = WS./vecnorm(WS); WX = WX./vecnorm(WX); WY = WY./vecnorm(WY);

S = sigmaS*randn(nS, N);

if poiss_on
    % Latent Gaussian population states
    
    X_lat = USX*VSX'*S + sigmaX*randn(nX, N);

    Y_lat = USY*VSY'*S + ...
            UXY*VXY'*X_lat + ...
            mod_stren*WY*((WS'*S).*(WX'*X_lat)) + ...
            sigmaY*randn(nY, N);

    X = poissrnd(exp(X_lat + baseline_h));
    Y = poissrnd(exp(Y_lat + baseline_h));

else
    X = USX*VSX'*S + sigmaX*randn(nX, N);

    Y = USY*VSY'*S + ...
        UXY*VXY'*X + ...
        mod_stren*WY*((WS'*S).*(WX'*X)) + ...
        sigmaY*randn(nY, N);
end

end


%% helpers
function theta = principal_angles(U, V)
% PRINCIPAL_ANGLES  Principal angles between column spaces of U and V.
% Returns k = min(rank(U), rank(V)) angles in radians.

    % Orthonormalize column spaces
    [QU, ~] = qr(U, 0);
    [QV, ~] = qr(V, 0);

    % Singular values of overlap
    s = svd(QU' * QV);

    % Clamp for numerical stability
    s = min(max(s, -1), 1);

    % Principal angles
    theta = acos(s);
end

function [U, V] = rank_factorization(A, k)
    % Ensure A has rank >= k
    [n, ~] = size(A);
    [U_svd, S, V_svd] = svd(A, 'econ');
    
    % Take the top k singular values/vectors
    U_k = U_svd(:, 1:k);
    S_k = S(1:k, 1:k);
    V_k = V_svd(:, 1:k);
    
    % Build U and V for rank factorization
    U = U_k * sqrt(S_k);
    V = V_k * sqrt(S_k);
end