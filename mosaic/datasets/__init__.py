
import os
import re
import cv2
import random
from pyquaternion import Quaternion
from mosaic.datasets.savers import Trajectory, HDF5Trajectory
import pickle as pkl
import numpy as np

from PIL import Image

SHUFFLE_RNG = 2843014334
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
SAWYER_DEMO_PRIOR = np.array([-1.95254033,  -3.25514605,  1.0,        -2.85691298,  -1.41135844,
                            -1.33966008,  -3.25514405,  1.0,        -2.89548721,  -1.26587143,
                            0.86143858,  -2.36652955,  1.0,        -2.61823206,   0.2176199,
                            -3.54059052,  -4.00911932,  1.0,        -5.07546054,  -5.25952708,
                            -3.7442406,   -5.35087854,  1.0,        -4.2814715,   -3.72755719,
                            -3.85309935,  -5.71775012,  1.0,        -7.02858012,  -3.15234408,
                            -3.67211177, 1.0,         -3.22493535, -12.45458803, -11.76144085, 0, 0])


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

def convert_angle_to_quat(angle_axis):
    angle = angle_axis[0] * np.pi
    if angle > 2 * np.pi:
        angle -= 2*np.pi
    axis = angle_axis[1:]
    quat = Quaternion(axis=axis, angle=angle)
    return quat

def convert_quat_to_angle(quat):
    a_qx, a_qy, a_qz, a_qw = quat
    quat = Quaternion(a_qw, a_qx, a_qy, a_qz)
    aa = np.concatenate(([quat.angle / np.pi], quat.axis)).astype(np.float32)
    if aa[0] < 0:
        aa[0] += 2
    return aa