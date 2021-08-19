MODEL=/home/mandi/mosaic/log_data/7Task-TempContra-Batch61-2gpu-Attn2ly128-Act2ly64mix7-headCat-simclr128x256
export CUDA_VISIBLE_DEVICES=4,5,6,7

for S in 4000 6000 8000 10000
do

for TASK in door button drawer stack_block pick_place nut_assembly basketball
do

python tasks/test_models/test_any_task.py $MODEL --wandb_log --env $TASK --saved_step $S

done
done
