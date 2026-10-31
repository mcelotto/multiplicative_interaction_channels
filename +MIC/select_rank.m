function opt_idx = select_rank(ve, sem_ve)
    ve_max = max(ve);
    sem_max = sem_ve(ve==ve_max); 
    if numel(sem_max)>1
        sem_max=sem_max(1); % if multiple models have the same EV, consider the simplest one
    end
    opt_idx = min(find((ve+sem_max)>=ve_max));
end
