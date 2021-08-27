TASK=PickPlace 
TASK_str=new_pick_place
python train_any.py policy='${mosaic}' exclude_task=${TASK_str} \
    exp_name=6Tasks-No${TASK} bsize=45 vsize=45 actions.n_mixtures=6 save_freq=2000 val_freq=2000


