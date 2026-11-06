%%%%%%%%%%%%%%%%%%%%%%%
% bilinear simulations used in Fig. S6 (MICs vs MLP)
% produces bilinear_saver.mat, loaded by figS6.m
%%%%%%%%%%%%%%%%%%%%%%%

results_path = @@ your path; % folder where bilinear_saver.mat is saved

% simulation settings

a_list = linspace(0, 1, 6); % modulation strength parameter
num_runs_per = 20;
N = 1e4;
nS = 5; nY = 5; % nX = 1 always here
mA_gt = 2;
% mC_gt = 1; % this is still what we're using, but it doesn't play an
% explicit part in the code
noise = 1;

% fitting setttings
params = struct;
params.max_lin_rank = 10;
params.K = 10;
params.ctgs_folds_on = true; 
params.seed = 42; 
params.opts = struct;
params.stratify_by = [];
params.max_mod_rank = 10;

results_all = cell(length(a_list), num_runs_per);

for jjj = 1:length(a_list)
    for kkk = 1:num_runs_per
        disp([jjj kkk])
        a = a_list(jjj);
        rng("shuffle");


        UA = randn(nS, mA_gt); VA = randn(nY, mA_gt); UA = UA./vecnorm(UA, 2, 1); VA = VA./vecnorm(VA, 2, 1);
        A_gt = UA*VA';

        % so alpha_ss = 1/2, alpha_sn = 1/2 is ground-truth
        WS_gt = randvec_angle_to_subspace(UA, 0);
        WY_gt = randvec_angle_to_subspace(VA, pi/4);
        WX_gt = 1;
        WS_gt = WS_gt./vecnorm(WS_gt, 2, 1); WY_gt = WY_gt./vecnorm(WY_gt, 2, 1);



        S = randn(N, nS);
        X = randn(N, 1);
        Y = S*A_gt + a*((S*WS_gt).*(X*WX_gt))*WY_gt.' + noise*randn(N, nY);

        results = MIC.model_wrapper(S, X, Y, params);


        abra = struct;
        abra.results = results;
        abra.S = S;
        abra.X = X;
        abra.Y = Y;
        abra.A_gt = A_gt;
        abra.WS_gt = WS_gt;
        abra.WY_gt = WY_gt;
        abra.WX_gt = WX_gt;
        results_all{jjj, kkk} = abra;
    end
end

saver = struct;
saver.results = results_all;
saver.params = params;
saver.a_list = a_list;
save(fullfile(results_path, "bilinear_saver"), "saver", "-v7.3");
