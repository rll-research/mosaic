import sys
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pickle as pkl
import imageio

for i in range(30):
    file = open('/data/fangchen/robosuite/new_place_model/results_pick_place/traj'+str(i)+'.pkl','rb')
    traj = pkl.load(file)
    out = imageio.get_writer('vis_place/out'+str(i)+'.gif')
    for i in range(traj.T):
        obs = traj.get(i)
        if 'obs' in obs:
            img = obs['obs']['image']
            out.append_data((img).astype(np.uint8))
    out.close()