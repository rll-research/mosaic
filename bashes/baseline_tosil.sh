TASK=Door
RUN_name=TOSIL
python train_blines.py policy='${tosil}' exp_name=1Task-${TASK}-TOSIL-lr5e4-Epoch10 epochs=10;
python train_blines.py policy='${tosil}' exp_name=1Task-${TASK}-TOSIL-lr1e4-Epoch10 epochs=10 train_cfg.lr=1e-4 
python train_blines.py policy='${tosil}' exp_name=1Task-${TASK}-TOSIL-lr3e4-Epoch10 epochs=10 train_cfg.lr=3e-4;