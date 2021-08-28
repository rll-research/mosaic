Task=Button
Task_name=button

for MODEL in \
    1Task-${Task}-TOSIL-lr5e4-Epoch10-Batch30 \
        1Task-${Task}-TOSIL-lr1e4-Epoch10-Batch30 \
            1Task-${Task}-TOSIL-lr3e4-Epoch10-Batch30
            do

            python tasks/test_models/test_one_model.py $MODEL --last_few 5  -bline 'tosil' --eval_tasks ${Task_name} --num_workers 2

            done

Task=Drawer
Task_name=drawer
for MODEL in \
    1Task-${Task}-TOSIL-lr5e4-Epoch10-Batch30 \
    1Task-${Task}-TOSIL-lr1e4-Epoch10-Batch30 \
    1Task-${Task}-TOSIL-lr3e4-Epoch10-Batch30
do

taskset -c 0-80 python tasks/test_models/test_one_model.py $MODEL --last_few 5  -bline 'tosil' --eval_tasks ${Task_name} --num_workers 2

done


