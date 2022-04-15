# pabamd2
taskset -c 0-80 python train_blines.py policy='${tosil}' use_all_tasks=True exp_name=7Task-${METHOD}-lr5e4-Epoch20-2lyAttn-mix7 epochs=20 tosil.n_mixtures=7 \
    set_same_n=1 bsize=61 vsize=61  save_freq=2000 val_freq=2000 single_task=False
# pabamd1
taskset -c 0-80 python train_blines.py policy='${tosil}' use_all_tasks=True exp_name=7Task-${METHOD}-lr5e4-Epoch20-3lyAttn-mix14 epochs=20 \
    tosil.n_mixtures=14 tosil.vis.n_st_attn=3 \
    set_same_n=1  bsize=61 vsize=61  save_freq=2000 val_freq=2000 single_task=False