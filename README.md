# mosaic
Code for Paper: Towards More Generalizable One-Shot Visual Imitation Learning

# Install 
system information
```
x86_64 GNU/Linux
Driver Version: 465.19.01    CUDA Version: 11.3 
```
package update:
```
sudo apt-get update
sudo apt-get --assume-yes install cmake libopenmpi-dev zlib1g-dev xvfb git g++ libfontconfig1 libxrender1 libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev xpra patchelf libglfw3-dev libglfw3 libglew2.0
```
- Install latest (free!) version of MuJoCo: this should create a folder under `.mujoco/mujoco210`

```
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz --no-check-certificate
tar -xf  mujoco210-linux-x86_64.tar.gz
```


- (Recommended) Create conda environment:
```
conda create -n mosaic python=3.7 pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=10.1 -c pytorch 
# python 3.8: 
conda create -n mosaic python=3.8 pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=11.3  -c pytorch

```
- Install `mujoco_py` and Robosuite v1.1:
see https://github.com/openai/mujoco-py for details on installing `mujoco_py`x
```
sudo mkdir -p /usr/lib/nvidia-000
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
pip install mujoco_py 
# install new robosuite and pybullet
pip install robosuite==1.1 # note we use an earlier version of robosuite
pip install pybullet==2.6.9
```
- To use pybullet IK solver for sawyer, update the URDF file
```
cd mosaic 
PATH_TO_CONDA_ENV=XXX # replace with local conda env path, e.g. PATH_TO_CONDA_ENV}=/home/mandi
cp tasks/robosuite_env/sawyer/sawyer_arm.urdf ${PATH_TO_CONDA_ENV}/miniconda3/envs/mosaic/lib/python3.7/site-packages/robosuite/models/assets/bullet_data/sawyer_description/urdf/sawyer_arm.urdf 
```

- Install additional packages and the benchmark task environment:
```
python setup.py develop
pip install -r requirements.txt
python tasks/setup.py develop
pip install -e tasks/.
```
