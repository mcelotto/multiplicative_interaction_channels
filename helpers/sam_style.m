function [] = sam_style()
% puts a matlab figure in the style I like
set(gca, 'FontSize', 12);
set(gca, 'LineWidth', 1.5);
box off;
set(gca, 'TickDir', 'out');
set(gca, 'TickLength', 0.01*[1, 1]);
set(get(gca,'XLabel'),'FontSize',16);
set(get(gca,'YLabel'),'FontSize',16);
set(gcf, 'Color', 'w');
end