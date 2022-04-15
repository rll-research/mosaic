# pabti5: try small model on 25, 50, 75% data
# not sure whether new pickplace needs a bigger model size, 
# try 2ly small for now
#------ small model scratch: ------
TASK=PickPlace
TASK_str=new_pick_place
EPOCH=36 
BSIZE=32
N=2
FEW=75
EXP_NAME=Limit${FEW}-1Task-${TASK}


TASK=PickPlace
TASK_str=new_pick_place
EPOCH=80
BSIZE=32
N=2
FEW=50
EXP_NAME=Limit${FEW}-1Task-${TASK}


TASK=PickPlace
TASK_str=new_pick_place
EPOCH=334
BSIZE=32
N=2
FEW=25
EXP_NAME=Limit${FEW}-1Task-${TASK}

python train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=${EXP_NAME}   \
      samplers.use_diy=True set_same_n=${N} bsize=${BSIZE} vsize=${BSIZE} \
      limit_num_demo=${FEW} limit_num_traj=${FEW}  \
      actions.n_mixtures=2 epochs=${EPOCH} save_freq=500 log_freq=500 val_freq=500 ; \
      MODEL=Limit${FEW}-1Task-${TASK}-Batch${BSIZE}-1gpu-Attn2ly128-Act2ly64mix2-headCat-simclr128x256 ; \
      python tasks/test_models/test_one_model.py $MODEL --last_few 1 --eval_tasks ${TASK_str} --num_workers 3

#------ bigger model sweeping, same naming as rtxs1 ------
TASK=PickPlace
TASK_str=new_pick_place
EPOCH=20 
BSIZE=32
N=2

python train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=1Task-${TASK}   \
      samplers.use_diy=True set_same_n=${N} bsize=${BSIZE} vsize=${BSIZE} \
      actions.n_mixtures=6 epochs=${EPOCH} save_freq=500 log_freq=500 val_freq=500 ; \
      MODEL=1Task-${TASK}-Batch${BSIZE}-1gpu-Attn2ly128-Act2ly128mix6-headCat-simclr128x256 ; \
      python tasks/test_models/test_one_model.py $MODEL --last_few 1 --eval_tasks ${TASK_str} --num_workers 3

python train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=1Task-${TASK}-NoIntm   \
      samplers.use_diy=True set_same_n=${N} bsize=${BSIZE} vsize=${BSIZE} simclr.mul_intm=0 \
      actions.n_mixtures=6 epochs=${EPOCH} save_freq=500 log_freq=500 val_freq=500 ; \
      MODEL=1Task-${TASK}-NoIntm-Batch${BSIZE}-1gpu-Attn2ly128-Act2ly128mix6-headCat-simclr128x256 ; \
      python tasks/test_models/test_one_model.py $MODEL --last_few 1 --eval_tasks ${TASK_str} --num_workers 3
