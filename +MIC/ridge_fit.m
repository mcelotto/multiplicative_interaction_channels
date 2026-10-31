function A = ridge_fit(X, Y, lambda) %+MIC
A = (X.'*X + eye(size(X, 2))*lambda)\(X.'*Y);
end