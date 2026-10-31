%%%%%%%%%%%%%%%%%%%%%%%%%
% code to generate fig S6
% further analyses of ACAxVIP data
%%%%%%%%%%%%%%%%%%%%%%%%%

load(@@ path to fig 4 ACAxVIP results); % --> (res, res_cstr, res_shuf, alpha_null_subtr, keep)
paths = @@ save paths

% plot results of shuffle test
ms = nan(length(res_shuf{1}.EV_mod_shuf{1}), 28);
as = nan(size(ms));
EV_A = nan(1, 28); EV_mod = nan(1, 28);
for i = 1:28
    ms(:, i) = (res_shuf{i}.EV_mod_shuf{1}+res_shuf{i}.EV_mod_shuf{2})/2; % null distribution of fold-avg mod EV
    as(:, i) = (res_shuf{i}.EV_A_shuf{1}+res_shuf{i}.EV_A_shuf{2})/2; % null distribution of fold-avg AIS EV
    EV_A(i) = mean(res_shuf{i}.EV_A); 
    EV_mod(i) = mean(res_shuf{i}.EV_mod);
end

figure;
subplot(3, 1, 2); hold on; % mod
boxplot(ms, 'Symbol', ''); % null
scatter(1:28, EV_mod); % actual

subplot(3, 1, 1); hold on; % AIS
boxplot(as, 'Symbol', ''); % null
scatter(1:28, EV_A); % actual

subplot(3, 1, 3); hold on; % p-values
scatter(1:28, res_shuf.p_A, 30, 'Marker', 's');
scatter(1:28, res_shuf.p_mod, 30, 'Marker', 'o');
set(gca, 'yscale', 'log');

%%% compute alignment indices
align_hat = nan(28, 2);
align_shuf = nan(28, 2);
num_shufs = 1e3; % number of random rotations
for i = 1:length(res)
    Ahat = mean(res{i}.cvA{res{i}.mA_try==res{i}.mA_opt}.A, 3);
    WShat = mean(res{i}.cv_mod{res{i}.mC_try==res{i}.mC_opt}.WS, 3);
    WYhat = mean(res{i}.cv_mod{res{i}.mC_try==res{i}.mC_opt}.WY, 3);
    angles = compute_angles(Ahat, res{i}.mA_opt, WShat, WYhat);
    al_idx.source = cos(deg2rad(angles.source)).^2;
    al_idx.target = cos(deg2rad(angles.target)).^2;
    align_hat(i, :) = [mean(al_idx.source) mean(al_idx.target)];

    temp = zeros(1,2);
    for j = 1:num_shufs
        angles = compute_angles(Ahat, res{i}.mA_opt, randn(size(WShat)), randn(size(WYhat)));
        al_idx.source = cos(deg2rad(angles.source)).^2;
        al_idx.target = cos(deg2rad(angles.target)).^2;
        temp = temp + [mean(al_idx.source) mean(al_idx.target)]/num_shufs;
    end
    align_shuf(i,:) = temp;
end

align_hat = align_hat(keep, :); align_shuf = align_shuf(keep, :);
align_idx = (align_hat-align_shuf)./(1-align_shuf);

%%% null substracted alpha indices vs population size imbalance
alphas_lab = {'ss','ns','sn','nn'};
for i = 1:numel(res)
    nACA(i) = res{i}.nS;
    nVIP(i) = res{i}.nY;
end
nACA = nACA(keep); nVIP = nVIP(keep);

fig = figure();
cols = [1,0,0; 0,1,0; 0,0,1; 0,0,0];
hold on
clear h
for i = 1:size(alpha_null_subtr,2)
    mu = alpha_null_subtr(:, i)';

    h(i) = scatter(nVIP./nACA,mu,'MarkerFaceColor',cols(i,:),'MarkerEdgeColor',cols(i,:));
    coeffs=polyfit(nVIP./nACA,mu,1);

    tmpX = 0:0.01:0.2;
    plot(tmpX,coeffs(2)+tmpX*coeffs(1),'color',cols(i,:))
    [c,p]=corr([nVIP./nACA]',mu');
    p = p*size(alpha_hat,2); % Bonferroni, adjusting p by N comparisons (here, 4)
    p = min([p,1]); % cap pval at 1
    pvalues_plot(p,0.15,0.95-0.15*i,0.05,1,14,0,cols(i,:),0)

end
ylim([-1,1])
legend(h,alphas_lab,'Location','NorthWest')
xlabel('VIP/ACA dim')
ylabel(['alpha shuff-sub'])
figure_save(paths.figures,'nullSub_alphas_w_popSizes',fig)

%%% plot AIS-MIC alignment indices

fig = figure();
hold on
xshift = linspace(-0.1,0.1,size(align_idx,1));
bar(1,mean(align_idx(:,1)),'FaceColor','r')
errorbar(1,mean(align_idx(:,1)),std(align_idx(:,1))/sqrt(size(align_idx,1)),'k','LineStyle','none')
scatter(1+xshift,align_idx(:,1),'MarkerFaceColor',[0.6,0.6,0.6],'MarkerEdgeColor',[0.6,0.6,0.6])

bar(2,mean(align_idx(:,2)),'FaceColor','b')
errorbar(2,mean(align_idx(:,2)),std(align_idx(:,2))/sqrt(size(align_idx,1)),'k','LineStyle','none')
scatter(2+xshift,align_idx(:,2),'MarkerFaceColor',[0.6,0.6,0.6],'MarkerEdgeColor',[0.6,0.6,0.6])
plot([1,2]'+xshift,align_idx','-','Color',[0.6,0.6,0.6])
ylabel('Align idx')
[~,p]=ttest(align_idx(:,1),align_idx(:,2),'Tail','Left');
pvalues_plot(p,1.5,0.9,0.05,1,18,0,'k',0)
xticks([1,2])
xticklabels({'ACA','VIP'})

figure_save(paths.figures,'align_idxs_bars',fig)

% alignment idx diff as function of size unbalance
fig=figure();
cols = [1,0,0;0,1,0;0,0,1;0 0 0];
% Shuff-subtracted plot
hold on
mu = squeeze(align_idx(:,2)-align_idx(:, 1)); mu = mu(:)';

h(i) = scatter(nVIP./nACA,mu,'MarkerFaceColor','k','MarkerEdgeColor','k');
coeffs=polyfit(nVIP./nACA,mu,1);

tmpX = 0:0.01:0.2;
plot(tmpX,coeffs(2)+tmpX*coeffs(1),'color','k')
[c,p]=corr([nVIP./nACA]',mu');
pvalues_plot(p,0.15,0.8-0.05*i,0.05,1,18,0,'k',0)
ylim([-1,1])
xlabel('VIP/ACA dim')
ylabel(['Align idx (VIP-ACA)'])

figure_save(paths.figures,'alignIdxs_w_popSize',fig)

%%% visualize alignment of AIS and MIC for an example session
keep_sess = find(keep);
sel_sess = 14;

plot_AIS_decomp(sel_sess,res{sel_sess},paths)




%%%%% HELPERS %%%%%
function [row_order,col_order,fig] = plot_AIS_decomp(sel_s, res, paths)
    
src_label = 'ACA';
targ_label = 'VIP';

[ACA, VIP, wheel, pupil] = readACAVIP(sel_s);

nTarget = size(VIP,2);

X = ACA; Y = VIP;
X = zscore(X,[],1); Y = zscore(Y,[],1);
Z = [wheel, pupil];
Z = (Z-mean(Z,'omitnan'))./std(Z,'omitnan');

keep = ~isnan(sum(X,2)) & ~isnan(sum(Y,2)) & ~isnan(sum(Z,2));
X = X(keep,:); Y = Y(keep,:); Z = Z(keep,:);

B_idx = find(res.mB_try == res.mB_opt);
subspaces.B = mean(res.cvB{B_idx}.B,3);
[subspaces.UB, subspaces.VB] = ...
    rank_factorization(subspaces.B, res.mB_opt);

D_idx = find(res.mD_try == res.mD_opt);
subspaces.D = mean(res.cvD{D_idx}.A,3); % D subspace is A due to indexing
[subspaces.UD, subspaces.VD] = ...
    rank_factorization(subspaces.D, res.mD_opt);

A_idx = find(res.mA_try == res.mA_opt);
subspaces.A = mean(res.cvA{A_idx}.A,3);
[subspaces.UA, subspaces.VA] = ...
    rank_factorization(subspaces.A, res.mA_opt);

C_idx = find(res.mC_try == res.mC_opt);
subspaces.WX = mean(res.cv_mod{C_idx}.WS,3); 
subspaces.WX = subspaces.WX./vecnorm(subspaces.WX);
subspaces.WZ = mean(res.cv_mod{C_idx}.WX,3); 
subspaces.WZ = subspaces.WZ./vecnorm(subspaces.WZ);
subspaces.WY = mean(res.cv_mod{C_idx}.WY,3); 
subspaces.WY = subspaces.WY./vecnorm(subspaces.WY);

% regress out the modulator
Y1 = Y - Z*subspaces.B;
X1 = X - (Z*subspaces.D);

src_targ_corr = corr(Y1,X1);

data = src_targ_corr;

% Hierarchical clustering to order row / columns according to
% similarity

% row order: treat each row as a feature vector
Dr = pdist(data, 'euclidean');  
Zr = linkage(Dr, 'ward');
row_order = optimalleaforder(Zr, Dr);

% column order: transpose and do the same
Dc = pdist(data', 'euclidean');
Zc = linkage(Dc, 'ward');
col_order = optimalleaforder(Zc, Dc);

% Reorder corr matrix, source and target communication subspaces
data = data(row_order, col_order);
UA = subspaces.UA(col_order,:);
VA = subspaces.VA(row_order,:);

sign_flip_UA = double(mean(UA,1)>0);
sign_flip_UA(sign_flip_UA==0) = -1;
UA(:,sign_flip_UA<0)=-UA(:,sign_flip_UA<0);
sign_flip_VA = double(mean(VA,1)>0);
sign_flip_VA(sign_flip_VA==0) = -1;
VA(:,sign_flip_VA<0)=-VA(:,sign_flip_VA<0);

% Define communication plotting colors
linestyles = {'-','--',':','-.'};  % MATLAB available styles
cmap = colormap("hot");
comm_cols = cmap(round(linspace(10,200,size(UA,2))),:); 

% Plot cross-correlation and subspaces
fig=figure('Position',[102,56,1075,636]);
subplot(3,3,[4,5,7,8])
imagesc(data)
xlabel(src_label)
ylabel(targ_label)
cmap = minmax_colormap_br(data);
colormap(cmap)
clim([-max(abs(data),[],'all'),max(abs(data),[],'all')]);

subplot(3,3,[1 2])
hold on
for i = 1:size(UA,2)
    plot(UA(:,i),'color',comm_cols(i,:),'LineWidth',2)
end
xlim([1,size(UA,1)])
xlabel(src_label)
ylabel('coeffs')
title('UA coefficients')
subplot(3,3,[6 9])
hold on
for i = 1:size(UA,2)
    plot(VA(:,i),1:nTarget,'color',comm_cols(i,:),'LineWidth',2)
end
ylim([1,size(VA,1)])
set(gca,'YDir','reverse')
xlabel('coeffs')
ylabel(targ_label)
title('VA coefficients')

figure_save(paths.figures,['examp',num2str(sel_s),'_comm_subspace_decomp'],fig)

% Plot reconstructed correlation matrix
figure()
data2 = VA*UA';
imagesc(data2)
xlabel(src_label)
ylabel(targ_label)
cmap = minmax_colormap_br(data2);
colormap(cmap)
clim([-max(abs(data2),[],'all'),max(abs(data2),[],'all')]);
title('Reconstructed cross-correlation matrix')

% Plot modulation and communication channels

W_src = subspaces.WX(col_order,:);
W_targ = subspaces.WY(row_order,:);

% Define modulation color
tmp = colormap("cool");
mod_cols = tmp(round(linspace(230,10,size(W_src,2))),:);

fig=figure();
data = subspaces;
assert(size(W_targ,2)<=numel(linestyles), ['Modulation rank (',targ_label,') exceeds number of linesyles'])
assert(size(UA,2)<=numel(linestyles), ['Communication rank (',targ_label,') exceeds number of linesyles'])
subplot(2,1,1)
hold on
sign_flip_mod = double(mean(W_src,1)>0);
sign_flip_mod(sign_flip_mod==0) = -1;
h1 = plot(sign_flip_mod(1)*W_src(:,1),'color',mod_cols(1,:),'LineWidth',2);
for i = 1:size(data.WZ,2)
    plot(sign_flip_mod(i)*W_src(:,i),'color',mod_cols(i,:),'LineWidth',2);
end
encoding_weigths_A = (UA);
ylabel('coefficient')
% yyaxis right
h2 = plot(encoding_weigths_A(:,1),'color',comm_cols(1,:),'LineWidth',2);
for i = 1:size(encoding_weigths_A,2)
    plot(encoding_weigths_A(:,i),'color',comm_cols(i,:),'LineWidth',2);
end
xlim([1,size(encoding_weigths_A,1)])
xlabel(src_label)
ylabel('coefficient')
legend([h1,h2],{['W_{ACA}'],'U_{A}'})

subplot(2,1,2)
hold on
h1 = plot(sign_flip_mod(1)*W_targ(:,1),'color',mod_cols(1,:),'LineWidth',2);
for i = 1:size(data.WX,2)
    plot(sign_flip_mod(i)*W_targ(:,i),'color',mod_cols(i,:),'LineWidth',2);
end
encoding_weigths_A = VA;
xlim([1,size(encoding_weigths_A,1)])
ylabel('coefficient')
% yyaxis right
h2 = plot(encoding_weigths_A(:,1),'color',comm_cols(1,:),'LineWidth',2);
for i = 1:size(encoding_weigths_A,2)
    plot(encoding_weigths_A(:,i),'color',comm_cols(i,:),'LineWidth',2);
end
xlabel(targ_label)
ylabel('coefficient')
legend([h1,h2],{['W_{VIP}'],'V_{A}'})

% Compute angles comm / mod
[U, ~, V] = svd(UA*VA', "econ"); % U output, V input (A is nSxnY)
U = U(:, 1:size(UA,2)); V = V(:, 1:size(VA,2));
PX = U*U'; PY = V*V';

for i = 1:size(W_src, 2)   
    % Compute angles in deg between mod channels and src / target comm
    % subspaces
    cos2_src = norm(W_src(:, i)'*PX, "fro")^2/norm(W_src(:,i), "fro")^2; % cos^2(theta_src)
    angles.source(i) = rad2deg(acos(sqrt(cos2_src)));
    cos2_targ = norm(PY*W_targ(:, i), "fro")^2/norm(W_targ(:,i), "fro")^2; % cos^2(theta_targ)
    angles.target(i) = rad2deg(acos(sqrt(cos2_targ)));

    for s = 1:100
         W_src_sh = randn(size(W_src));
         W_targ_sh = randn(size(W_targ));
         cos2_src_sh(s) = norm(W_src_sh(:, i)'*PX, "fro")^2/norm(W_src_sh(:,i), "fro")^2; % cos^2(theta_src)
         angles_shuf.source(i,s) = rad2deg(acos(sqrt(cos2_src_sh(s))));
         cos2_targ_sh(s) = norm(PY*W_targ_sh(:, i), "fro")^2/norm(W_targ_sh(:,i), "fro")^2; % cos^2(theta_targ)
         angles_shuf.target(i,s) = rad2deg(acos(sqrt(cos2_targ_sh(s))));
    end
    align_score.source(i) = (cos2_src-mean(cos2_src_sh))/(1-mean(cos2_src_sh));
    align_score.target(i) = (cos2_targ-mean(cos2_targ_sh))/(1-mean(cos2_targ_sh));
end

subplot(2,1,1)
title([targ_label,' target - align idxs: ',num2str(align_score.source,2)])
subplot(2,1,2)
title([targ_label,' target - align idxs: ',num2str(align_score.target,2)])
figure_save(paths.figures,['examp',num2str(sel_s),'_comm_subspace_mod_channels'],fig)

% Pairwise correlation (abs value)
fig=figure('Position',[149,142,882,487]);
for i = 1:size(W_src,2)
    
    subplot(2,2,1)
    c = UA \ W_src(:,i);                 % least-squares coefficients
    y_hat = UA * c;                     % best linear combination

    hold on
    plot(y_hat,'r','LineWidth',2,'LineStyle',linestyles{i})
    plot(W_src(:,i),'k','LineWidth',2,'LineStyle',linestyles{i})
    xlim([1,numel(W_src(:,i))])
    minY = min([min([y_hat,W_src(:,i)]),0]);
    maxY = max([max([y_hat,W_src(:,i)]),0]);
    ylim([minY, maxY])
    ylabel('coefficients')
    xlabel(src_label) 
    title(src_label)
    legend({'Best comm subspace direction','Modulation channel'},'Location','SouthWest')

    subplot(2,2,3)
    hold on
    plot(W_src(:,i)-y_hat,'LineWidth',2,'LineStyle',linestyles{i})
    title('difference')
    xlim([1,numel(W_src(:,i))])
    ylim([-0.3,0.3])
    ylabel('coefficients')
    xlabel(src_label) 
    title(src_label)

    subplot(2,2,2)

    c = VA \ W_targ(:,i);                         % least-squares coefficients
    y_hat = VA * c;                     % best linear combination

    hold on
    plot(y_hat,'r','LineWidth',2,'LineStyle',linestyles{i})
    plot(W_targ(:,i),'k','LineWidth',2,'LineStyle',linestyles{i})
    xlim([1,numel(W_targ(:,i))])
    minY = min([min([y_hat;W_targ(:,i)]),0]);
    maxY = max([max([y_hat;W_targ(:,i)]),0]);
    ylim([minY, maxY])
    ylabel('coefficients')
    xlabel(targ_label) 
    title(targ_label)

    subplot(2,2,4)
    hold on
    plot(W_targ(:,i)-y_hat,'LineWidth',2,'LineStyle',linestyles{i})
    title('difference')
    xlim([1,numel(W_targ(:,i))])
    ylim([-0.3,0.3])
    ylabel('coeff difference')
    xlabel(targ_label) 
    title(targ_label)
end
figure_save(paths.figures,['examp',num2str(sel_s),'_comm_subspace_mod_channels_reconstruction'],fig)

end

function [U, V] = rank_factorization(A, k)
    % Ensure A has rank >= k
    [n, ~] = size(A);
    [U_svd, S, V_svd] = svd(A, 'econ');
    
    % Take the top k singular values/vectors
    U_k = U_svd(:, 1:k);
    S_k = S(1:k, 1:k);
    V_k = V_svd(:, 1:k);
    
    U = U_k; V = V_k;
    % Build U and V for rank factorization
    % U = U_k * sqrt(S_k);
    % V = V_k * sqrt(S_k);
end


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

% Compute angles in deg between mod channels and src / target comm subspaces
function angles = compute_angles(A, mA, WS, WY)
    [U, ~, V] = svd(A', "econ"); % U output, V input (A is nSxnY)
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
