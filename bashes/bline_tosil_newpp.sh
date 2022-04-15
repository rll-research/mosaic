TASK=PickPlaceNew
METHOD=TOSIL
python train_blines.py policy='${tosil}' single_task='${new_pick_place}' exp_name=1Task-${TASK}-${METHOD}-lr5e4-Epoch20 epochs=20 
#python train_blines.py policy='${tosil}' single_task='${new_pick_place}'  exp_name=1Task-${TASK}-${METHOD}-lr1e4-Epoch10 epochs=10 train_cfg.lr=1e-4 ;
#python train_blines.py policy='${tosil}' single_task='${new_pick_place}'  exp_name=1Task-${TASK}-${METHOD}-lr3e4-Epoch10 epochs=10 train_cfg.lr=3e-4 ;
