# [rtxs1]
# 6 task:
TASK=PickPlace 
TASK_str=new_pick_place
python train_any.py policy='${mosaic}' exclude_task=${TASK_str} \
    exp_name=6Tasks-No${Task} bsize=45 vsize=45 actions.n_mixtures=6

TASK=PickPlace 
TASK_str=new_pick_place
python train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=1Task-${TASK} \
    samplers.use_diy=False simclr.compressor_dim=256 simclr.hidden_dim=512

python tasks/test_models/test_one_model.py $MODEL --last_few 5 --eval_tasks ${TASK_str} --num_workers 3


TASK=StackBlock
TASK_str=stack_block 
python train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=1Task-${TASK} \
    samplers.use_diy=False 
python train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=1Task-${TASK} \
    samplers.use_diy=True set_same_n=5 bsize=30 vsize=30 actions.n_mixtures=2

python train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=1Task-${TASK}     samplers.use_diy=True set_same_n=5 bsize=30 vsize=30 actions.n_mixtures=2 epochs=20 ; MODEL=1Task-StackBlock-Batch30-1gpu-Attn2ly128-Act2ly64mix2-headCat-simclr128x256 ; python tasks/test_models/test_one_model.py $MODEL --last_few 5 --eval_tasks ${TASK_str} --num_workers 3

# [pabamd1] # mix4+batch30 not reproducing results
TASK=NutAssembly
TASK_str=nut_assembly
taskset -c 0-80 python train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=1Task-${TASK} \
    samplers.use_diy=False
TASK=NutAssembly
TASK_str=nut_assembly
taskset -c 0-80 python train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=1Task-${TASK} \
    samplers.use_diy=True set_same_n=3 actions.n_mixtures=2 bsize=27 vsize=27

# [pabti5]
TASK=Basketball
TASK_str=basketball
python train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=1Task-${TASK} \
    samplers.use_diy=False  

TASK=Basketball
TASK_str=basketball
python train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=1Task-${TASK} \
    samplers.use_diy=True set_same_n=3 actions.n_mixtures=2 bsize=36 vsize=36

# [pabti1] 6task, bigger model
TASK=PickPlace 
TASK_str=new_pick_place
python train_any.py policy='${mosaic}' exclude_task=${TASK_str} \
    exp_name=6Tasks-No${TASK} actions.n_mixtures=6 \
    samplers.use_diy=1 set_same_n=2 bsize=90 vsize=90 \
    save_freq=2000 val_freq=2000 actions.n_mixtures=6 attn.attn_ff=256