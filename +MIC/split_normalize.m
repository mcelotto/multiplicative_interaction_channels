function [Xtr, Xte] = split_normalize(X, tr, te)
% train-test split
Xtr = X(tr, :);
Xte = X(te, :);
    
% scale test and train set by train set mean & SD
Xte = (Xte - mean(Xtr, 1))./std(Xtr, [], 1);
Xtr = (Xtr - mean(Xtr, 1))./std(Xtr, [], 1);

end

