function [V1, V2, stim] = readSemedo(idx, params)

% Reads one session of the V1-V2 dataset of Zandvakili & Kohn (2015), as
% analyzed in Semedo et al. (2019) (CRCNS, https://crcns.org/data-sets/vc/v1v2-1).
%
% Inputs:
%   idx      session index (1-5, alphabetical order of the session files)
%   params   struct with fields:
%     tbin        spike count bin width (ms)
%     stim_code   'one_hot': one-hot encoding of the 8 orientations
%                 '2dim':    (sin(2*theta), cos(2*theta))
%
% Outputs:
%   V1    [nV1 x T]  V1 spike counts (PSTH subtracted)
%   V2    [nV2 x T]  V2 spike counts (PSTH subtracted)
%   stim  [8 x T] or [2 x T]  orientation encoding

folder = @@ path to the CRCNS v1-v2 'mat_neural_data' folder;

% list session files
files = dir(fullfile(folder, '**', '*.*'));
keep = true(size(files));
bad = [".", "..", ".DS_Store"];
for i = 1:length(files)
    fullName = fullfile(files(i).folder, files(i).name);
    if ismember(files(i).name, bad) | isdir(fullName)
        keep(i) = false;
    end
end
files = files(keep);
abc = load(fullfile(files(idx).folder, files(idx).name));

% orientation (deg) of each stimulus code, per session
stdOri = [0 22.5 45 67.5 90 112.5 135 157.5];
oriBySession = containers.Map( ...
    {'105l001p16', '106r001p26', '106r002p70', '107l002p67', '107l003p143'}, ...
    {[0 22.5 45 67.5 90 112.5 180 157.5], stdOri, stdOri, stdOri, stdOri});
[~, stem] = fileparts(files(idx).name);
oriVec = oriBySession(stem);

nTrials = numel(abc.neuralData.stim); % stimulus presentations alternate with blank periods (stim = 0)
nV1 = size(abc.neuralData.unitCodes{1}, 1);
nV2 = size(abc.neuralData.unitCodes{2}, 1);

% analysis window (ms after stimulus onset), as in Semedo et al. (2019)
tWinStart = 160; tWinEnd = 1160;
tbins_on = floor((tWinEnd-tWinStart)/params.tbin);
nStim = sum(abc.neuralData.stim > 0);
tot_tPoints = tbins_on * nStim;

V1 = nan(nV1, tot_tPoints);
V2 = nan(nV2, tot_tPoints);
oriCol = nan(1, tot_tPoints); % stimulus code of each bin
binCol = nan(1, tot_tPoints); % within-trial bin index of each bin

switch params.stim_code
    case '2dim'
        stim = nan(2, tot_tPoints);
    case 'one_hot'
        stim = nan(8, tot_tPoints);
end

% bin spike counts of stimulus periods
binS = 1;
for t = 1:nTrials
    s = abc.neuralData.stim(t);
    if s == 0 % skip blank periods
        continue;
    end

    data1 = abc.neuralData.spikeRasters{t, 1};
    data2 = abc.neuralData.spikeRasters{t, 2};

    % restrict to the analysis window
    data1 = data1(:, (tWinStart+1):tWinEnd);
    data2 = data2(:, (tWinStart+1):tWinEnd);

    [~, nT] = size(data1); % rasters stored in ms; same length for V1 and V2
    nBins = ceil(nT / params.tbin);
    edges = [(0:nBins-1)'*params.tbin, min((1:nBins)'*params.tbin, nT)];
    exclIdx = find(mod(edges(:, 2), params.tbin)>0);

    % remove time points of incomplete bins
    if exclIdx
        nT = edges(exclIdx, 1);
        data1(:, nT+1:end) = [];
        data2(:, nT+1:end) = [];
        edges(exclIdx, :) = [];
        nBins = ceil(nT / params.tbin);
    end
    colIdx = ceil((1:nT) / params.tbin);
    B = sparse(1:nT, colIdx, 1, nT, nBins);

    binE = binS+nBins-1;
    V1(:, binS:binE) = full(data1 * B);
    V2(:, binS:binE) = full(data2 * B);

    oriCol(binS:binE) = s;
    binCol(binS:binE) = 1:nBins;

    switch params.stim_code
        case '2dim'
            theta = deg2rad(oriVec(s));
            stim(:, binS:binE) = repelem([sin(2*theta); cos(2*theta)], 1, nBins);
        case 'one_hot'
            tmp = zeros(1, 8)';
            tmp(s) = 1;
            stim(:, binS:binE) = repelem(tmp, 1, nBins);
    end

    binS = binE+1;
end

% exclude units firing below 0.5 spikes/s
secTotal = size(V1, 2) * params.tbin / 1000;
keepV1 = (sum(V1, 2) / secTotal) >= 0.5;
keepV2 = (sum(V2, 2) / secTotal) >= 0.5;
V1 = V1(keepV1, :);
V2 = V2(keepV2, :);

% subtract PSTH: per unit, orientation, and within-trial bin
for s = unique(oriCol)
    for b = 1:tbins_on
        m = (oriCol==s) & (binCol==b);
        V1(:, m) = V1(:, m) - mean(V1(:, m), 2);
        V2(:, m) = V2(:, m) - mean(V2(:, m), 2);
    end
end

end
