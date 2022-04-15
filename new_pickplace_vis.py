import os 
#os.environ['LD_LIBRARY_PATH'] = '/home/mandi/.mujoco/mujoco200/bin:/usr/lib/nvidia-430'
from robosuite import load_controller_config
import json
import matplotlib.pyplot as plt
import numpy as np
from robosuite_env.controllers.expert_pick_place import \
    get_expert_trajectory as place_expert 
import random 
create_seed = random.Random(None)
create_seed = create_seed.getrandbits(32)
controller = load_controller_config(default_controller='IK_POSE')

robot = 'Panda'
heights=100
widths=180
controller = load_controller_config(default_controller='IK_POSE')
traj = get_expert_trajectory('{}PickPlaceDistractor'.format(robot), heights=heights, widths=widths,
    controller_type=controller, task=2, ret_env=1, renderer=False, seed=create_seed)
obs = traj.reset()
img = obs['image']
a,b,c,d = [10,50,70,70]
img = img[a:-b, c:-d, :]
plt.imshow(img)
plt.axis('off')
plt.savefig('PickPlace.png')
