function alpha = compute_alphas(A, mA, WS, WY)
    % Computes fractional geometry indices (Eq. 10) for each MIC, quantifying
    % how the source and target weight vectors align with the baseline AIS.
    %
    % NOTE ON VARIABLE NAMING: WS and WY correspond to WX and WY in the paper
    % (source and target MIC factor matrices, respectively).
    %
    % Inputs:
    %   A    [nS x nY]   Baseline AIS matrix (paper: A)
    %   mA   scalar      Rank of baseline AIS (number of singular vectors retained)
    %   WS   [nS x mC]   Source MIC weights; columns are {wX,i} (paper: WX)
    %   WY   [nY x mC]   Target MIC weights; columns are {wY,i} (paper: WY)
    %
    % Outputs:
    %   alpha   Struct with fields [1 x mC], one value per MIC channel:
    %     .in    alpha_ss: fraction of channel strength with source AND target
    %                      in the baseline AIS
    %     .Yrot  alpha_sn: fraction with source in AIS, target outside AIS
    %     .Srot  alpha_ns: fraction with source outside AIS, target in AIS
    %     .new   alpha_nn: fraction with source AND target outside AIS
    
    [U, ~, V] = svd(A', "econ"); % U output, V input (A is nSxnY)
    U = U(:, 1:mA); V = V(:, 1:mA);
    PY = U*U'; PS = V*V';
    alpha = struct;
    for i = 1:size(WS, 2)
        deltaA = WY(:, i)*WS(:, i)';
        alpha.in(i) = norm(PY*deltaA*PS, "fro")^2/norm(deltaA, "fro")^2;
        alpha.Yrot(i) = norm((eye(size(PY))-PY)*deltaA*PS, "fro")^2/norm(deltaA, "fro")^2;
        alpha.Srot(i) = norm(PY*deltaA*(eye(size(PS))-PS), "fro")^2/norm(deltaA, "fro")^2;
        alpha.new(i) = norm((eye(size(PY))-PY)*deltaA*(eye(size(PS))-PS), "fro")^2/norm(deltaA, "fro")^2;
    end
end
