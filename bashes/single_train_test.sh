# train:
TASK=StackBlock
TASK_str=stack_block
EPOCH=20
BSIZE=30
N=5
python train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=1Task-${TASK}   \
      set_same_n=${N} bsize=${BSIZE} vsize=${BSIZE} \
      actions.n_mixtures=2 actions.out_dim=64 attn.attn_ff=128  simclr.mul_intm=0  \
      simclr.compressor_dim=128 simclr.hidden_dim=256 \
      epochs=${EPOCH} save_freq=1000 log_freq=500 val_freq=1000  

# test:
MODEL=1Task-StackBlock-Batch30-1gpu-Attn2ly128-Act2ly64mix2-headCat-simclr128x256
python tasks/test_models/test_one_model.py $MODEL --last_few 1 --eval_tasks ${TASK_str} --num_workers 3

 
TASK_str=basketball
EPOCH=30
BSIZE=36
N=3

 
TASK=Basketball
TASK_str=basketball
EPOCH=30
BSIZE=36
N=3 

TASK=Drawer
TASK_str=drawer
EPOCH=3
BSIZE=32
N=4

TASK=Door
TASK_str=door
EPOCH=15
BSIZE=32
N=8


TASK=Button
TASK_str=button
EPOCH=10
BSIZE=30
N=5 

TASK=NutAssembly
TASK_str=nut_assembly
EPOCH=20 
BSIZE=27
python train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=1Task-${TASK}   \
      bsize=${BSIZE} vsize=${BSIZE} \
      actions.n_mixtures=2 actions.out_dim=64 attn.attn_ff=128  simclr.mul_intm=0  \
      simclr.compressor_dim=128 simclr.hidden_dim=256 \
      epochs=${EPOCH} save_freq=1000 log_freq=500 val_freq=1000 ; \



TASK=PickPlace
TASK_str=new_pick_place
EPOCH=20 
BSIZE=32
N=2

python train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=1Task-${TASK}   \
      samplers.use_diy=True set_same_n=${N} bsize=${BSIZE} vsize=${BSIZE} \
      actions.n_mixtures=6 actions.out_dim=128 attn.attn_ff=256 simclr.compressor_dim=256 simclr.hidden_dim=512 \
      epochs=${EPOCH} save_freq=500 log_freq=500 val_freq=500 ; \
      MODEL=1Task-${TASK}-Batch${BSIZE}-1gpu-Attn2ly256-Act2ly128mix6-headCat-simclr256x512 ; \
      python tasks/test_models/test_one_model.py $MODEL --last_few 1 --eval_tasks ${TASK_str} --num_workers 3
 