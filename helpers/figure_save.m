function figure_save(path,filename,figname)

if ~exist(path,'dir')
    mkdir(path)
end

tmp_filename = [path,'\',filename,'.png'];
saveas(figname,tmp_filename)

tmp_filename = [path,'\',filename,'.fig'];
saveas(figname,tmp_filename)

tmp_filename = [path,'\',filename,'.svg'];
saveas(figname,tmp_filename)

print(figname,'-painters','-bestfit',[path,'\',filename],'-dpdf')

end