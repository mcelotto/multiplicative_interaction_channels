function EV = compute_EV(Ytrain, Ytest, Ypred)
    R_model = Ytest - Ypred;
    R_base = Ytest - mean(Ytrain, 1);

    SSE = sum(R_model(:).^2);
    SST = sum(R_base(:).^2);

    EV = 1 - SSE / SST;
end