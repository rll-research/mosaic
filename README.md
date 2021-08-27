# mosaic
Code for Paper: Towards More Generalizable One-Shot Visual Imitation Learning

# Install 

```
sudo apt-get update
sudo apt-get --assume-yes install cmake libopenmpi-dev zlib1g-dev xvfb git g++ libfontconfig1 libxrender1 libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev xpra patchelf libglfw3-dev libglfw3 libglew2.0
```
- (Recommended) Create conda environment:
```
conda create -n mosaic python=3.7 pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=10.1 -c pytorch 
[trying pabrtxs3:]
conda create -n mosaic python=3.8 pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=11.3  -c pytorch 

```
- Obtain mujoco license and install `mujoco200` first, then
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
PATH_TO_CONDA_ENV=XXX # replace with local conda env path
PATH_TO_CONDA_ENV=/home/mandi
cp tasks/robosuite_env/sawyer/sawyer_arm.urdf ${PATH_TO_CONDA_ENV}/miniconda3/envs/mosaic/lib/python3.7/site-packages/robosuite/models/assets/bullet_data/sawyer_description/urdf/sawyer_arm.urdf 
```

- Install additional packages
```
python setup.py develop
pip install -r requirements.txt
python tasks/setup.py develop
pip install -e tasks/.
```
