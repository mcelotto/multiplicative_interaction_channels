function [ACA, VIP, wheel, pupil, stim_on, t, type] = readACAVIP(idx)

% Reads one session of the ACA-VIP dataset (https://osf.io/2ekj4/).
% ACA axon and VIP neuron traces are linearly detrended.
%
% Inputs:
%   idx      session index (1-28, alphabetical order of the session files)
%
% Outputs:
%   ACA      [T x nACA]  ACA axon traces
%   VIP      [T x nVIP]  VIP neuron traces
%   wheel    [T x 1]     running speed
%   pupil    [T x 1]     pupil diameter
%   stim_on  [T x 1]     true during stimulus presentation
%   t        [T x 1]     timestamps
%   type     'gratings' or 'movies'

folder = @@ path to the ACA-VIP csv folder;

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
path = fullfile(files(idx).folder, files(idx).name);
abc = readtable(path);
tab_names = abc.Properties.VariableNames;

if contains(path, 'gratings')
    type = 'gratings';
elseif contains(path, 'movies')
    type = 'movies';
else
    type = nan;
end

ACA = [];
i = 1;
while sum(strcmp(['axon' num2str(i)], tab_names))==1
    idx = find(strcmp(['axon' num2str(i)], tab_names));
    ACA(:, i) = abc{:, idx};
    i = i + 1;
end

VIP = [];
i = 1;
while sum(strcmp(['vip' num2str(i)], tab_names))==1
    idx = find(strcmp(['vip' num2str(i)], tab_names));
    VIP(:, i) = abc{:, idx};
    i = i + 1;
end

% remove linear trend
xx = (1:size(ACA, 1))';
for i = 1:size(ACA, 2)
    p = polyfit(xx, ACA(:, i), 1);
    ACA(:, i) = ACA(:, i) - (p(1)*xx + p(2));
end

xx = (1:size(VIP, 1))';
for i = 1:size(VIP, 2)
    p = polyfit(xx, VIP(:, i), 1);
    VIP(:, i) = VIP(:, i) - (p(1)*xx + p(2));
end

% arousal measurements
wheel = abc.speed;
pupil = abc.pupil_diameter;

% timestamps
t = abc.timestamp;

% stimulus periods
stim_on = ~isnan(abc.direction_movie);
end
