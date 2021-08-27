TASK=PickPlace 
TASK_str=new_pick_place
python train_any.py policy='${mosaic}' exclude_task=${TASK_str} \
    exp_name=6Tasks-No${Task} bsize=45 vsize=45 actions.n_mixtures=6