# Towards More Generalizable One-Shot Visual Imitation Learning
ArXiv: https://arxiv.org/abs/2110.13423

## Installation
Note that the below instructions are mainly tested on Linux x86_64 with NVIDIA GPUs (`Driver Version: 465.19.01 CUDA Version: 11.3`) 
1. package update:
```
sudo apt-get update
sudo apt-get --assume-yes install cmake libopenmpi-dev zlib1g-dev xvfb git g++ libfontconfig1 libxrender1 libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev xpra patchelf libglfw3-dev libglfw3 libglew2.0
```
2. Install latest version of MuJoCo and move to path `.mujoco/mujoco210`

```
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz --no-check-certificate
tar -xf  mujoco210-linux-x86_64.tar.gz
```
3. (Recommended) Create conda environment:
```
conda create -n mosaic python=3.7 pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=10.1 -c pytorch 
conda activate mosaic
```
4. Install `mujoco_py` and Robosuite v1.1:

see https://github.com/openai/mujoco-py for more detailed instructions
```
sudo mkdir -p /usr/lib/nvidia-000
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
pip install mujoco_py 
# install this specific version of robosuite and pybullet
pip install robosuite==1.1 # note we use an earlier version of robosuite
pip install pybullet==2.6.9
```
5. To use pybullet IK solver for sawyer, update the URDF file
```
cd mosaic 
PATH_TO_CONDA_ENV=XXX # replace with local conda env path, e.g. PATH_TO_CONDA_ENV}=/home/mandi
cp tasks/robosuite_env/sawyer/sawyer_arm.urdf ${PATH_TO_CONDA_ENV}/miniconda3/envs/mosaic/lib/python3.7/site-packages/robosuite/models/assets/bullet_data/sawyer_description/urdf/sawyer_arm.urdf 
```

6. Install the remaining packages and the benchmark task environment:
```
python setup.py develop
pip install -r requirements.txt
python tasks/setup.py develop
pip install -e tasks/.
```
## Generating Expert Data
1. Specify desired path to dataset:
```
SUITE=${PATH_TO_DATA}/mosaic_multitask_dataset
```
2. Example of generating data for a single task, `Press Button`
```
TASK_name=button
N_VARS=6 # number of variations for this task
NUM=600
for ROBOT in panda sawyer
do 
python tasks/collect_data/collect_any_task.py ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
  -tsk ${TASK_name} -ro ${ROBOT} --n_tasks ${N_VARS}  --n_env 800 \
   --N ${NUM} --per_task_group 100 --num_workers 20 --collect_cam  --heights 100 --widths 180
done 
```


## Model Training 
1. training a single task agent:
```
EXP_NAME=1Task-StackBlock
TASK_str=stack_block
EPOCH=20
BSIZE=30
python train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=${EXP_NAME}   \
      bsize=${BSIZE} actions.n_mixtures=2 actions.out_dim=64 attn.attn_ff=128  simclr.mul_intm=0  \
      simclr.compressor_dim=128 simclr.hidden_dim=256 epochs=${EPOCH}
```
## Testing Trained Models
1. test the last saved model checkpoint:
```
TASK_str=stack_block
MODEL=1Task-StackBlock-Batch30-1gpu-Attn2ly128-Act2ly64mix2-headCat-simclr128x256
python tasks/test_models/test_one_model.py $MODEL --last_few 1 --eval_tasks ${TASK_str} --num_workers 3
```
