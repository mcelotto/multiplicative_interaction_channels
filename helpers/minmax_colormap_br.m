function cmap = minmax_colormap_br(M)
    % Input: Matrix M 
    % Output: Colormap ranging from red to blue, mapping 0 value in M to white
 
    % Define colors
    b = [0 0 1]; % Blue
    r = [1 0 0]; % Red
    w = [1 1 1]; % White

    % Generate colormap
    n = 256; % Number of color levels

    % Find the percentile position of white_val in the range
    white_pos = 0.5;

    % Create a gradient from red through white to blue
    cmap_neg = interp1([0, white_pos], [b; w], linspace(0, white_pos, round(n * white_pos)));
    cmap_pos = interp1([white_pos, 1], [w; r], linspace(white_pos, 1, n - round(n * white_pos)));

    % Combine the two halves, avoiding repeating the white color
    cmap = [cmap_neg; cmap_pos(2:end, :)];
end