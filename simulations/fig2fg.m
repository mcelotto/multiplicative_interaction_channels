%%%%%%%%%%%%%%%%%%%%%%%
% simulations shown in Fig. 2f-g
% testing rank selection
%%%%%%%%%%%%%%%%%%%%%%%

% simulation settings
nS = 10; nY = 10; nX = 10; % population dimensionalities
N = 2000; % number of samples per sim
num_sims_per = 5; % number of sims per (mA, mC) pair
gt_rank_rng = 0:5; % list of mA, mC values to use
noise = 1; % emission noise
mod_stren = 1; % strength of MIC

% shuffle test settings
shuffle_type = "perm";
nshuffle = 1e3;
Kouter = 2;

% ground-truth ranks
mAl = [];
mCl = [];
for i = 1:length(gt_rank_rng)
    for j = 1:length(gt_rank_rng)
        mCl = [mCl ones(1, num_sims_per)*gt_rank_rng(i)];
        mAl = [mAl ones(1, num_sims_per)*gt_rank_rng(j)];
    end
end

% fitting parameters
params = struct;
params.K = 10;
params.ctgs_folds_on = false; 
params.seed = 42; 
params.opts = struct;
params.max_mod_rank = 10;
params.stratify_by = [];
params.max_lin_rank = 20;

% store results here
results_all = cell(size(mAl));

% iterate over ground-truth ranks
for aaa = 1:length(mAl)
    disp(aaa); rng("shuffle");

    mAgt = mAl(aaa); mCgt = mCl(aaa);

    % randomly oriented AIS
    UAgt = randn(nS, mAgt); UAgt = UAgt./vecnorm(UAgt, 2, 1);
    VAgt = randn(nY, mAgt); VAgt = VAgt./vecnorm(VAgt, 2, 1);
    Agt = UAgt*VAgt';

    % randomly oriented MIC
    WSgt = randn(nS, mCgt); WSgt = mod_stren*WSgt./vecnorm(WSgt, 2, 1);
    WYgt = randn(nY, mCgt); WYgt = WYgt./vecnorm(WYgt, 2, 1);
    WXgt = randn(nX, mCgt); WXgt = WXgt./vecnorm(WXgt, 2, 1);

    % simulate
    S = randn(N, nS);
    X = randn(N, nX);
    Y = S*Agt + ((S*WSgt).*(X*WXgt))*WYgt.' + noise*randn(N, nY);

    % fit
    all_data_model = MIC.model_wrapper(S, X, Y, params);

    % shuffle test on heldout data
    [inner_model, EV_A, EV_B, EV_D, EV_mod, EV_A_shuf, EV_B_shuf, EV_D_shuf, EV_mod_shuf, p_A, p_B, p_D, p_mod] = MIC.heldout_test(S, X, Y, params, Kouter, nshuffle, shuffle_type);

    % store simulated data and fitting and shuffle test results
    abc = struct; abc.S = S; abc.X = X; abc.Y = Y; abc.all_data_model = all_data_model; 
    abc.inner_model = inner_model; abc.EV_A = EV_A; abc.EV_mod = EV_mod; abc.EV_A_shuf = EV_A_shuf; abc.EV_mod_shuf = EV_mod_shuf; abc.p_A = p_A; abc.p_mod = p_mod;
    results_all{aaa} = abc;
end

mA = []; mC = []; pA = []; pC = [];
for i = 1:length(mAl)
    mA(i) = results_all{i}.all_data_model.mA_opt;
    mC(i) = results_all{i}.all_data_model.mC_opt;
    pA(i) = results_all{i}.p_A;
    pC(i) = results_all{i}.p_mod;
end

% if shuffle test failed to reject null hypothesis, set corresponding rank
% to 0
sig_level = 0.05;
mA(pA>=sig_level) = 0;
mC(pC>=sig_level) = 0;

% rank selection confusion matrices
bins = 0:5;
Aconf = zeros(length(bins)+1, length(bins));
Cconf = zeros(length(bins)+1, length(bins));

for j = 1:length(mA)
    r = find(bins==mA(j));
    c = find(bins==mAl(j));
    
    if length(r)>0
        Aconf(r, c) = Aconf(r, c)+1;
    else
        Aconf(end, c) = Aconf(end, c)+1;
    end
    
    r = find(bins==mC(j));
    c = find(bins==mCl(j));
    
    if length(r)>0
        Cconf(r, c) = Cconf(r, c)+1;
    else
        Cconf(end, c) = Cconf(end, c)+1;
    end

end

% plot
figure;

% confusion matrices
subplot(1, 4, 3);
heatmap(Cconf);

subplot(1, 4, 4);
heatmap(Aconf);

% example rank-selection curves
subplot(1, 4, 1);
errorbar(0:min(nS, nY), results_all{idx}.all_data_model.veA, results_all{idx}.all_data_mode.semA, 'k', 'LineWidth', 1.5);

subplot(1, 4, 2);
errorbar(0:params.max_mod_rank, results_all{idx}.all_data_model.ve_mod, results_all{idx}.all_data_mode.sem_mod, 'k', 'LineWidth', 1.5);