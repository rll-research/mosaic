# all on pabrtxs3
METHOD=DAML
TASK=Button
TASK_str='${button}'
taskset -c 0-80 python train_blines.py use_maml=True  single_task=${TASK_str} exp_name=1Task-${TASK}-${METHOD}-lr0.1-Epoch10 epochs=10;
# python train_blines.py policy='${tosil}' single_task=${TASK_str}  exp_name=1Task-${TASK}-${METHOD}-lr1e4-Epoch10 epochs=10 train_cfg.lr=1e-4 ;
python train_blines.py policy='${tosil}' single_task=${TASK_str}  exp_name=1Task-${TASK}-${METHOD}-lr3e4-Epoch10 epochs=10 train_cfg.lr=3e-4 ;

TASK=Basketball
TASK_str='${basketball}'
python train_blines.py policy='${tosil}' single_task=${TASK_str} exp_name=1Task-${TASK}-${METHOD}-lr5e4-Epoch10 epochs=10;
python train_blines.py policy='${tosil}' single_task=${TASK_str}  exp_name=1Task-${TASK}-${METHOD}-lr1e4-Epoch10 epochs=10 train_cfg.lr=1e-4 ;
python train_blines.py policy='${tosil}' single_task=${TASK_str}  exp_name=1Task-${TASK}-${METHOD}-lr3e4-Epoch10 epochs=10 train_cfg.lr=3e-4 ;