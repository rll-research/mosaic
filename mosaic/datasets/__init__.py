from mosaic.datasets.savers.trajectory import Trajectory
from mosaic.datasets.savers.hdf5_trajectory import HDF5Trajectory
import pickle as pkl
import random 
import glob
import os
import re
SHUFFLE_RNG = 2843014334
try:
    raise NotImplementedError
    from mosaic.datasets.savers.render_loader import ImageRenderWrapper
    import_render_wrapper = True
except:
    import_render_wrapper = False

def load_traj(fname):
    if '.pkl' in fname:
        traj = pkl.load(open(fname, 'rb'))['traj']
    elif '.hdf5' in fname:
        traj = HDF5Trajectory()
        traj.load(fname)
    else:
        raise NotImplementedError

    traj = traj if not import_render_wrapper else ImageRenderWrapper(traj)
    return traj


def split_files(file_len, splits, mode='train'):
    assert sum(splits) == 1 and all([0 <= s for s in splits]), "splits is not valid pdf!"

    order = [i for i in range(file_len)]
    random.Random(SHUFFLE_RNG).shuffle(order)
    pivot = int(len(order) * splits[0])
    if mode == 'train':
        order = order[:pivot]
    else:
        order = order[pivot:]
    return order

