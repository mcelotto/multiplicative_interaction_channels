%%%%%%%%%%%%%%%%%%%%%%%
% simulations shown in Fig. S1
% pipeline validation across all 16 rank combinations
%%%%%%%%%%%%%%%%%%%%%%%

% simulation settings
nS = 5; nY = 5; nX = 5; % pop dimensionalities
N = 1000; % num samples per sim
noise = 1; % emission noise
mod_stren = 1; % MIC strength
num_sims = 15; % num sims per condition

% heldout shuffle test settings
nshuffle = 1000;
shuffle_type = "perm";
Kouter = 2;

% fitting parameters
params = struct;
params.K = 10;
params.ctgs_folds_on = false; 
params.seed = 42;
params.opts = struct;
params.max_mod_rank = 5;
params.stratify_by = [];
params.max_lin_rank = 5;

% 16 scenarios (all the combinations of the four terms being on vs. off)
vals = [0 1];
[mA_all, mB_all, mC_all, mD_all] = ndgrid(vals, vals, vals, vals);
rank_sets = [mA_all(:), mB_all(:), mC_all(:), mD_all(:)];

% store results here
res = cell(size(rank_sets, 1), num_sims, 2); % regress out, no sig test
sigtest = cell(size(res)); % regress out, sig test
res_no_regress_out = cell(size(rank_sets, 1), num_sims, 2); % no regress out, no sig test 
sigtest_no_regress_out = cell(size(res_no_regress_out)); % no regress out, sig test
gt_and_data = cell(size(rank_sets, 1), num_sims, 2); % ground truth models and simulated data

% iterate over scenarios
for aa = 1:size(rank_sets, 1)
    % ground-truth ranks
    mAgt = rank_sets(aa, 1); mBgt = rank_sets(aa, 2); mCgt = rank_sets(aa, 3); mDgt = rank_sets(aa, 4);

    % iterate over independent sims
    for k = 1:num_sims
        % if two or more of mB, mC, mD are turned on, simulate both "rand"
        % and "corr" scenarios; otherwise simulate only "rand" scenario
        if mBgt+mCgt+mDgt>1; l_list = 1:2; else; l_list = 1; end
        for l = l_list
            disp([aa, k, l]); rng("shuffle");

            % randomly oriented ground-truth AIS
            UAgt = randn(nS, mAgt); UAgt = UAgt./vecnorm(UAgt, 2, 1);
            VAgt = randn(nY, mAgt); VAgt = VAgt./vecnorm(VAgt, 2, 1);
            Agt = UAgt*VAgt';

            if l == 2
                % "corr" scenario
                Svec = randn(nS, 1); Svec = Svec/vecnorm(Svec, 2);
                Xvec = randn(nS, 1); Xvec = Xvec/vecnorm(Xvec, 2);
                Yvec = randn(nS, 1); Yvec = Yvec/vecnorm(Yvec, 2);

                if mBgt==1
                    Bgt = Xvec*Yvec';
                else
                    Bgt = zeros(nX, nY);
                end

                if mCgt==1
                    WSgt = mod_stren*Svec; WXgt = Xvec; WYgt = Yvec;
                else
                    WSgt = 0*Svec; WXgt = 0*Xvec; WYgt = 0*Yvec;
                end

                if mDgt==1
                    Dgt = Xvec*Svec';
                else
                    Dgt = zeros(nX, nS);
                end
            else
                % "rand" scenario
                UBgt = randn(nX, mBgt); UBgt = UBgt./vecnorm(UBgt, 2, 1);
                VBgt = randn(nY, mBgt); VBgt = VBgt./vecnorm(VBgt, 2, 1);
                Bgt = UBgt*VBgt';

                UDgt = randn(nX, mDgt); UDgt = UDgt./vecnorm(UDgt, 2, 1);
                VDgt = randn(nS, mDgt); VDgt = VDgt./vecnorm(VDgt, 2, 1);
                Dgt = UDgt*VDgt';

                WSgt = randn(nS, mCgt); WSgt = mod_stren*WSgt./vecnorm(WSgt, 2, 1);
                WYgt = randn(nY, mCgt); WYgt = WYgt./vecnorm(WYgt, 2, 1);
                WXgt = randn(nX, mCgt); WXgt = WXgt./vecnorm(WXgt, 2, 1);
            end

            % simulate
            X = randn(N, nX);
            S = X*Dgt + noise*randn(N, nS);
            Y = X*Bgt + S*Agt + ((S*WSgt).*(X*WXgt))*WYgt.' + noise*randn(N, nY);

            abc = struct;
            abc.Agt = Agt; abc.Bgt = Bgt; abc.Dgt = Dgt;
            abc.WSgt = WSgt; abc.WXgt = WXgt; abc.WYgt = WYgt;
            abc.X = X; abc.S = S; abc.Y = Y;
            gt_and_data{aa, k, l} = abc;

            % regress out
            % run on full dataset
            res{aa, k, l} = MIC.model_wrapper(S, X, Y, params);

            % heldout data sig test
            [inner_model, EV_A, EV_B, EV_D, EV_mod, EV_A_shuf, EV_B_shuf, EV_D_shuf, EV_mod_shuf, p_A, p_B, p_D, p_mod] = MIC.heldout_test(S, X, Y, params, Kouter, nshuffle, shuffle_type);
            htest = struct; htest.inner_model = inner_model; htest.EV = {EV_A, EV_B, EV_mod, EV_D}; htest.EVshuf = {EV_A_shuf, EV_B_shuf, EV_mod_shuf, EV_D_shuf}; htest.p = [p_A p_B p_mod, p_D];
            sigtest{aa, k, l} = htest;

            % don't regress out
            % run on full dataset
            params2 = params; params2.mB = 0; params2.mD = 0;
            res_no_regress_out{aa, k, l} = MIC.model_wrapper(S, X, Y, params);

            % heldout data sig test
            [inner_model, EV_A, EV_B, EV_D, EV_mod, EV_A_shuf, EV_B_shuf, EV_D_shuf, EV_mod_shuf, p_A, p_B, p_D, p_mod] = MIC.heldout_test(S, X, Y, params2, Kouter, nshuffle, shuffle_type);
            htest = struct; htest.inner_model = inner_model; htest.EV = {EV_A, EV_B, EV_mod, EV_D}; htest.EVshuf = {EV_A_shuf, EV_B_shuf, EV_mod_shuf, EV_D_shuf}; htest.p = [p_A p_B p_mod, p_D];
            sigtest_no_regress_out{aa, k, l} = htest;
        end
    end
end



pcut = 0.05; % significance level

ifr = nan([size(res) 4]); % inferred ranks (regress out, no sig test)
ifr_nro = nan([size(res_no_regress_out) 4]); % inferred ranks (no regress out, no sig test)

ifr_p = nan([size(res) 4]);  % inferred ranks (regress out, sig test)
ifr_nro_p = nan([size(res_no_regress_out) 4]); % inferred ranks (no regress out, sig test)

p = nan([size(sigtest) 4]); % p-values (regress out)
p_nro = nan([size(sigtest_no_regress_out) 4]); % p-values (no regress out)
for i = 1:size(res, 1)
    for j = 1:size(res, 2)
        for k = 1:size(res, 3)
            if ~isempty(res{i, j, k})
                c = res{i, j, k};
                ifr(i, j, k, :) = [c.mA_opt c.mB_opt c.mC_opt c.mD_opt];

                c = res_no_regress_out{i, j, k};
                ifr_nro(i, j, k, :) = [c.mA_opt c.mB_opt c.mC_opt c.mD_opt];

                c = sigtest{i, j, k};
                p(i, j, k, :) = c.p;

                c = sigtest_no_regress_out{i, j, k};
                p_nro(i, j, k, :) = c.p;

                ifr_p(i, j, k, :) = (p(i, j, k, :)<0.05).*ifr(i, j, k, :);
                ifr_nro_p(i, j, k, :) = (p_nro(i, j, k, :)<0.05).*ifr_nro(i, j, k, :);
                
            end
        end
    end
end

% compute rank selection accuracies
acc = nan([size(ifr, 1) size(ifr, 3) 5]); %5th entry is overall acc
acc_p = nan(size(acc));

acc_nro = nan(size(acc));
acc_nro_p = nan(size(acc));

for i = 1:size(ifr, 1)
    for j = 1:size(ifr, 3)
        for k = 1:size(ifr, 4)
            if ~isnan(ifr(i, 1, j, k))
                acc(i, j, k) = sum(ifr(i, :, j, k)==rank_sets(i, k))/size(ifr, 2);
                acc_p(i, j, k) = sum(ifr_p(i, :, j, k)==rank_sets(i, k))/size(ifr_p, 2);
                
                acc_nro(i, j, k) = sum(ifr_nro(i, :, j, k)==rank_sets(i, k))/size(ifr_nro, 2);
                acc_nro_p(i, j, k) = sum(ifr_nro_p(i, :, j, k)==rank_sets(i, k))/size(ifr_nro_p, 2);
                
            end
        end

        if ~isnan(ifr(i, 1, j, 1))
            % overall acc
            acc(i, j, 5) = sum(ismember(squeeze(ifr(i, :, j, :)), rank_sets(i, :), 'rows'))/size(ifr, 2);
            acc_p(i, j, 5) = sum(ismember(squeeze(ifr_p(i, :, j, :)), rank_sets(i, :), 'rows'))/size(ifr_p, 2);
            
            acc_nro(i, j, 5) = sum(ismember(squeeze(ifr_nro(i, :, j, :)), rank_sets(i, :), 'rows'))/size(ifr_nro, 2);
            acc_nro_p(i, j, 5) = sum(ismember(squeeze(ifr_nro_p(i, :, j, :)), rank_sets(i, :), 'rows'))/size(ifr_nro_p, 2);
        end
    end
end

% compute alphas, ground truth and estimated (for all of the different pipelines)

alpha_gt = nan(size(ifr)); % ground truth
alpha = nan(size(ifr)); 
alpha_p = nan(size(ifr));
alpha_nro = nan(size(ifr));
alpha_nro_p = nan(size(ifr));
for i = 1:size(ifr, 1)
    for j = 1:size(ifr, 2)
        for k = 1:size(ifr, 3)
            if sum(isnan(ifr(i, j, k, :)))==0
                if rank_sets(i, 3)>0
                    Agt = gt_and_data{i, j, k}.Agt; WSgt = gt_and_data{i, j, k}.WSgt; WYgt = gt_and_data{i, j, k}.WYgt;
                    abc = MIC.compute_alphas(Agt, rank_sets(i, 1), WSgt, WYgt);
                    alpha_gt(i, j, k, :) = [abc.in abc.Srot abc.Yrot abc.new];
                end
    
                c = res{i, j, k};
                if c.mC_opt>0
                    Ahat = mean(c.cvA{c.mA_opt==c.mA_try}.A, 3); 
                    WShat = mean(c.cv_mod{c.mC_opt==c.mC_try}.WS, 3);
                    WYhat = mean(c.cv_mod{c.mC_opt==c.mC_try}.WY, 3);
                    abc = MIC.compute_alphas(Ahat, c.mA_opt, WShat, WYhat);
                    alpha(i, j, k, :) = [mean(abc.in) mean(abc.Srot) mean(abc.Yrot) mean(abc.new)];
    
                    if p(i, j, k, 3)<pcut
                        mAtemp = c.mA_opt;
                        if p(i, j, k, 1)>=pcut
                            Ahat = zeros(size(Ahat)); 
                            mAtemp = 0; 
                        end
                        abc = MIC.compute_alphas(Ahat, mAtemp, WShat, WYhat);
                        alpha_p(i, j, k, :) = [mean(abc.in) mean(abc.Srot) mean(abc.Yrot) mean(abc.new)];
                    end
                end
    
                c = res_no_regress_out{i, j, k};
                if c.mC_opt>0
                    Ahat = mean(c.cvA{c.mA_opt==c.mA_try}.A, 3); 
                    WShat = mean(c.cv_mod{c.mC_opt==c.mC_try}.WS, 3);
                    WYhat = mean(c.cv_mod{c.mC_opt==c.mC_try}.WY, 3);
                    abc = MIC.compute_alphas(Ahat, c.mA_opt, WShat, WYhat);
                    alpha_nro(i, j, k, :) = [mean(abc.in) mean(abc.Srot) mean(abc.Yrot) mean(abc.new)];
    
                    if p_nro(i, j, k, 3)<pcut
                        mAtemp = c.mA_opt;
                        if p_nro(i, j, k, 1)>=pcut
                            Ahat = zeros(size(Ahat)); 
                            mAtemp = 0; 
                        end
                        abc = MIC.compute_alphas(Ahat, mAtemp, WShat, WYhat);
                        alpha_nro_p(i, j, k, :) = [mean(abc.in) mean(abc.Srot) mean(abc.Yrot) mean(abc.new)];
                    end
                end
            end
        end
    end
end

%%%%%%%%
% plot %
%%%%%%%%

% barplot spacing
sp = 0.2*(1:3);
ww = 0.75;

figure; 
subplot(9, 1, 2);
hold on; 
cols = rand(16, 3);
for i = 1:size(acc, 1) 
    ae = nanmean(abs(squeeze(alpha_p(i, :, 1, :))-squeeze(alpha_gt(i, :, 1, :))), 'all');
    h = bar(i+sp, [squeeze(acc_p(i, 1, [1 3])) ; ae], 'BarWidth', ww, 'LineWidth', 1, 'FaceColor', 'flat');
    h.CData = [repmat(cols(i,:), 2, 1); 0.5 0.5 0.5];
end

subplot(9, 1, 4);
hold on; 
for i = 1:size(acc_nro_p, 1) 
    ae = nanmean(abs(squeeze(alpha_nro_p(i, :, 1, :))-squeeze(alpha_gt(i, :, 1, :))), 'all');
    h = bar(i+sp, [squeeze(acc_nro_p(i, 1, [1 3])) ; ae], 'BarWidth', ww, 'LineWidth', 1, 'FaceColor', 'flat');
    h.CData = [repmat(cols(i,:), 2, 1); 0.5 0.5 0.5];
end

subplot(9, 1, 3);
hold on; 
for i = 1:size(acc, 1) 
    ae = nanmean(abs(squeeze(alpha_p(i, :, 2, :))-squeeze(alpha_gt(i, :, 2, :))), 'all');
    h = bar(i+sp, [squeeze(acc_p(i, 2, [1 3])) ; ae], 'BarWidth', ww, 'LineWidth', 1, 'FaceColor', 'flat');
    h.CData = [repmat(cols(i,:), 2, 1); 0.5 0.5 0.5];
end

subplot(9, 1, 5);
hold on; 
for i = 1:size(acc_nro_p, 1) 
    ae = nanmean(abs(squeeze(alpha_nro_p(i, :, 2, :))-squeeze(alpha_gt(i, :, 2, :))), 'all');
    h = bar(i+sp, [squeeze(acc_nro_p(i, 2, [1 3])) ; ae], 'BarWidth', ww, 'LineWidth', 1, 'FaceColor', 'flat');
    h.CData = [repmat(cols(i,:), 2, 1); 0.5 0.5 0.5];
end

subplot(9, 1, 6);
hold on; 
for i = 1:size(acc, 1) 
    ae = nanmean(abs(squeeze(alpha(i, :, 1, :))-squeeze(alpha_gt(i, :, 1, :))), 'all');
    h = bar(i+sp, [squeeze(acc(i, 1, [1 3])) ; ae], 'BarWidth', ww, 'LineWidth', 1, 'FaceColor', 'flat');
    h.CData = [repmat(cols(i,:), 2, 1); 0.5 0.5 0.5];
end

subplot(9, 1, 8);
hold on; 
for i = 1:size(acc_nro_p, 1) 
    ae = nanmean(abs(squeeze(alpha_nro(i, :, 1, :))-squeeze(alpha_gt(i, :, 1, :))), 'all');
    h = bar(i+sp, [squeeze(acc_nro(i, 1, [1 3])) ; ae], 'BarWidth', ww, 'LineWidth', 1, 'FaceColor', 'flat');
    h.CData = [repmat(cols(i,:), 2, 1); 0.5 0.5 0.5];
end

subplot(9, 1, 7);
hold on; 
for i = 1:size(acc, 1) 
    ae = nanmean(abs(squeeze(alpha(i, :, 2, :))-squeeze(alpha_gt(i, :, 2, :))), 'all');
    h = bar(i+sp, [squeeze(acc(i, 2, [1 3])) ; ae], 'BarWidth', ww, 'LineWidth', 1, 'FaceColor', 'flat');
    h.CData = [repmat(cols(i,:), 2, 1); 0.5 0.5 0.5];
end

subplot(9, 1, 9);
hold on; 
for i = 1:size(acc_nro_p, 1) 
    ae = nanmean(abs(squeeze(alpha_nro(i, :, 2, :))-squeeze(alpha_gt(i, :, 2, :))), 'all');
    h = bar(i+sp, [squeeze(acc_nro(i, 2, [1 3])) ; ae], 'BarWidth', ww, 'LineWidth', 1, 'FaceColor', 'flat');
    h.CData = [repmat(cols(i,:), 2, 1); 0.5 0.5 0.5];
end

for i = 2:9; subplot(9, 1, i); ylim([0 1]); yticks([0 1]); xticks([]); xlim([0 17+sp(end)+(sp(2)-sp(1))*ww/2]); end

subplot(9, 1, 1);
heatmap(rank_sets')

