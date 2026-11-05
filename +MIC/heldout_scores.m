function [sX, sZ, sY, foldid] = heldout_scores(results, S, X, Y, chan) % +MIC

% Held-out scores of one fitted MIC, used by bilinearity_test.m. In each
% CV fold, the test data are projected onto the channel weights and
% additive terms (A, B, D) fitted on the training data.
%
% NOTE ON VARIABLE NAMING: S, X, Y here correspond to X, Z, Y in the paper
% (source, modulator, target). The output scores follow the paper.
%
% Inputs:
%   results   model_wrapper output (selected ranks are used)
%   S, X, Y   data the model was fitted on, with NaN samples removed
%             (model_wrapper removes them before building the CV folds)
%   chan      index of the MIC
%
% Outputs:
%   sX      [N x 1]  residualized source score:      (S - X*D)*wX
%   sZ      [N x 1]  modulator score:                X*wZ
%   sY      [N x 1]  modulation residual score:      (Y - X*B - (S - X*D)*A)*wY
%   foldid  [N x 1]  CV fold of each sample

ia = find(results.mA_try == results.mA_opt, 1);
ib = find(results.mB_try == results.mB_opt, 1);
id = find(results.mD_try == results.mD_opt, 1);
ic = find(results.mC_try == results.mC_opt, 1);

cvp = results.cv_mod{ic}.cv_partition;
N = size(S, 1);
if cvp.NumObservations ~= N
    error('heldout_scores: CV partition has %d samples but data have %d rows (NaN removal differs from the fit)', ...
        cvp.NumObservations, N);
end

sX = nan(N, 1); sZ = nan(N, 1); sY = nan(N, 1); foldid = zeros(N, 1);
for l = 1:cvp.NumTestSets
    tr = training(cvp, l);
    te = test(cvp, l);
    foldid(te) = l;

    [~, Ste] = MIC.split_normalize(S, tr, te);
    [~, Xte] = MIC.split_normalize(X, tr, te);
    [~, Yte] = MIC.split_normalize(Y, tr, te);

    % additive terms (B is stored in field B by earlier code versions, in field A otherwise)
    Al = results.cvA{ia}.A(:, :, l);
    if isfield(results.cvB{ib}, 'B')
        Bl = results.cvB{ib}.B(:, :, l);
    else
        Bl = results.cvB{ib}.A(:, :, l);
    end
    Dl = results.cvD{id}.A(:, :, l);

    sr = Ste - Xte*Dl; % residualized source
    ym = (Yte - Xte*Bl) - sr*Al; % residual of the additive model

    WSl = results.cv_mod{ic}.WS(:, :, l);
    WXl = results.cv_mod{ic}.WX(:, :, l);
    WYl = results.cv_mod{ic}.WY(:, :, l);
    sX(te) = sr*WSl(:, chan);
    sZ(te) = Xte*WXl(:, chan);
    sY(te) = ym*WYl(:, chan);
end
end
