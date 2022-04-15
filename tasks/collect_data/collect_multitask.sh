SUITE=/home/mandi/mosaic_multitask_dataset
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
WORKERS=20
CPUS=0-36
SCRIPT=tasks/collect_data/collect_any_task.py

HEIGHT=100
WIDTH=180
N_env=800 
per_task=100

# button 
# TASK_name=button
# N_tasks=6
# NUM=600
# for ROBOT in panda sawyer
# do 
# taskset -c ${CPUS} python ${SCRIPT} ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
#         -tsk ${TASK_name} -ro ${ROBOT} --n_tasks ${N_tasks}  --n_env ${N_env} \
#         --N ${NUM} --per_task_group ${per_task} \
#         --num_workers ${WORKERS} --collect_cam \
#         --heights ${HEIGHT} --widths ${WIDTH} \
#         --overwrite # danger!!
# done 

# TASK_name=door
# N_tasks=4
# NUM=400
# for ROBOT in panda sawyer
# do 
# taskset -c ${CPUS} python ${SCRIPT} ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
#         -tsk ${TASK_name} -ro ${ROBOT} --n_tasks ${N_tasks}  --n_env ${N_env} \
#         --N ${NUM} --per_task_group ${per_task} \
#         --num_workers ${WORKERS} --collect_cam \
#         --heights ${HEIGHT} --widths ${WIDTH} \
#         --overwrite # danger!!
# done 


# TASK_name=drawer
# N_tasks=8
# NUM=800

# for ROBOT in panda sawyer
# do 
# taskset -c ${CPUS} python ${SCRIPT} ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
#         -tsk ${TASK_name} -ro ${ROBOT} --n_tasks ${N_tasks}  --n_env ${N_env} \
#         --N ${NUM} --per_task_group ${per_task} \
#         --num_workers ${WORKERS} --collect_cam \
#         --heights ${HEIGHT} --widths ${WIDTH} \
#         --overwrite # danger!!
# done 
 

# TASK_name=basketball
# N_tasks=12
# NUM=1200
# for ROBOT in panda sawyer
# do 
# taskset -c ${CPUS} python ${SCRIPT} ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
#         -tsk ${TASK_name} -ro ${ROBOT} --n_tasks ${N_tasks}  --n_env ${N_env} \
#         --N ${NUM} --per_task_group ${per_task} \
#         --num_workers ${WORKERS} --collect_cam \
#         --heights ${HEIGHT} --widths ${WIDTH} \
#         --overwrite # danger!!
# done 


# TASK_name=nut_assembly
# N_tasks=9
# NUM=900
# for ROBOT in panda sawyer
# do 
# taskset -c ${CPUS} python ${SCRIPT} ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
#         -tsk ${TASK_name} -ro ${ROBOT} --n_tasks ${N_tasks}  --n_env ${N_env} \
#         --N ${NUM} --per_task_group ${per_task} \
#         --num_workers ${WORKERS} --collect_cam \
#         --heights ${HEIGHT} --widths ${WIDTH} \
#         --overwrite # danger!!
# done 


# TASK_name=stack_block
# N_tasks=6
# NUM=600
# for ROBOT in panda sawyer
# do 
# taskset -c ${CPUS} python ${SCRIPT} ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
#         -tsk ${TASK_name} -ro ${ROBOT} --n_tasks ${N_tasks}  --n_env ${N_env} \
#         --N ${NUM} --per_task_group ${per_task} \
#         --num_workers ${WORKERS} --collect_cam \
#         --heights ${HEIGHT} --widths ${WIDTH} \
#         --overwrite # danger!!
# done 

TASK_name=pick_place  ## NOTE different size
N_tasks=16
NUM=16 #00
for ROBOT in panda # sawyer
do 
taskset -c ${CPUS} python ${SCRIPT} ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
        -tsk ${TASK_name} -ro ${ROBOT} --n_tasks ${N_tasks}  --n_env ${N_env} \
        --N ${NUM} --per_task_group ${per_task} \
        --num_workers ${WORKERS} --collect_cam \
        --heights 100 --widths 180 \
        --overwrite # danger!!
done 


# TASK_name=stack_new_color
# N_tasks=6
# NUM=600
# for ROBOT in panda sawyer
# do 
# taskset -c ${CPUS} python ${SCRIPT} ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
#         -tsk ${TASK_name} -ro ${ROBOT} --n_tasks ${N_tasks}  --n_env ${N_env} \
#         --N ${NUM} --per_task_group ${per_task} \
#         --num_workers ${WORKERS} --collect_cam \
#         --heights ${HEIGHT} --widths ${WIDTH} \
#         --overwrite --color # danger!!
# done 

# TASK_name=stack_new_shape 
# N_tasks=6
# NUM=600
# for ROBOT in panda sawyer
# do 
# taskset -c ${CPUS} python ${SCRIPT} ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
#         -tsk ${TASK_name} -ro ${ROBOT} --n_tasks ${N_tasks}  --n_env ${N_env} \
#         --N ${NUM} --per_task_group ${per_task} \
#         --num_workers ${WORKERS} --collect_cam \
#         --heights ${HEIGHT} --widths ${WIDTH} \
#         --overwrite --shape # danger!!
# done 
