for MODEL in /home/mandi/mosaic/log_data/7Task-TempContra-Batch61-2gpu-Attn2ly128-Act2ly64mix7-headCat-simclr128x256 7Task-TempContra-Batch61-2gpu-Attn2ly256-Act2ly64mix7-headCat-simclr128x256
#export CUDA_VISIBLE_DEVICES=4,5,6,7

do
for S in 90000 100000 120000
do

for TASK in door button drawer stack_block pick_place nut_assembly basketball
do

python tasks/test_models/test_any_task.py $MODEL --wandb_log --env $TASK --saved_step $S --eval_each_task 10
done

done
done
