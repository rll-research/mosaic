export CUDA_VISIBLE_DEVICES=0,1,2,5
Task=StackBlock
Task_name=stack_block

for MODEL in \
    1Task-${Task}-TOSIL-lr5e4-Epoch10-Batch30 \
        1Task-${Task}-TOSIL-lr1e4-Epoch10-Batch30 \
            1Task-${Task}-TOSIL-lr3e4-Epoch10-Batch30
            do

            python tasks/test_models/test_one_model.py $MODEL --last_few 5  -bline 'tosil' --eval_tasks ${Task_name} --num_workers 2

            done

Task=NutAssembly
Task_name=nut_assembly
for MODEL in \
    1Task-${Task}-TOSIL-lr5e4-Epoch10-Batch30 \
    1Task-${Task}-TOSIL-lr1e4-Epoch10-Batch30 \
    1Task-${Task}-TOSIL-lr3e4-Epoch10-Batch30
do

python tasks/test_models/test_one_model.py $MODEL --last_few 5  -bline 'tosil' --eval_tasks ${Task_name} --num_workers 2

done

Task=Basketball 
Task_name=basketball
for MODEL in \
    1Task-${Task}-TOSIL-lr5e4-Epoch10-Batch30 \
    1Task-${Task}-TOSIL-lr1e4-Epoch10-Batch30 \
    1Task-${Task}-TOSIL-lr3e4-Epoch10-Batch30
do

python tasks/test_models/test_one_model.py $MODEL --last_few 5  -bline 'tosil' --eval_tasks ${Task_name} --num_workers 2

done