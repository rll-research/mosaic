from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2axisangle
from robosuite_env.controllers.expert_basketball import \
    get_expert_trajectory as basketball_expert
from robosuite_env.controllers.expert_nut_assembly import \
    get_expert_trajectory as nut_expert
from robosuite_env.controllers.expert_pick_place import \
    get_expert_trajectory as place_expert 
from robosuite_env.controllers.expert_block_stacking import \
    get_expert_trajectory as stack_expert 
from robosuite_env.controllers.expert_drawer import \
    get_expert_trajectory as draw_expert 
from robosuite_env.controllers.expert_button import \
    get_expert_trajectory as press_expert 
from robosuite_env.controllers.expert_door import \
    get_expert_trajectory as door_expert 
from eval_functions import *
from hem.datasets.util import STD, MEAN, resize, crop, convert_angle_to_quat

import random
import copy
import os
from collections import defaultdict
from pyquaternion import Quaternion
from hem.util import parse_basic_config
import torch
from hem.datasets import Trajectory
import numpy as np
import pickle as pkl
import imageio
import functools
from torch.multiprocessing import Pool, set_start_method
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import cv2
import random

set_start_method('forkserver', force=True)
import hydra
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from torchvision.transforms import RandomAffine, ToTensor, Normalize
from torchvision.transforms import functional as TvF
from torchvision.transforms.functional import resized_crop
import matplotlib.pyplot as plt

TASK_MAP = {
    'bask': {
        'n_task':   4, 
        'env_fn':   basketball_expert,
        'eval_fn':  basketball_eval,
        'agent-teacher': ('PandaBasketball', 'SawyerBasketball'),
        'render_hw': (100, 180),
        },
    'bask_hard': {
        'n_task':   12, 
        'env_fn':   basketball_expert,
        'eval_fn':  basketball_eval,
        'agent-teacher': ('PandaBasketball', 'SawyerBasketball'),
        'render_hw': (150, 270), # new bask_hard is 150 270!
        },

    'nut':  {
        'n_task':   9, 
        'env_fn':   nut_expert,
        'eval_fn':  nut_assembly_eval,
        'agent-teacher': ('PandaNutAssemblyDistractor', 'SawyerNutAssemblyDistractor'),
        'render_hw': (100, 180), 
        },
    'nut_hard': {
        'n_task':   9,
        'env_fn':   nut_expert,
        'eval_fn':  nut_assembly_eval,
        'agent-teacher': ('PandaNutAssemblyDistractor', 'SawyerNutAssemblyDistractor'),
        'render_hw': (150, 270), # (180, 320)??? 0424 harder nut version 
        },
    'place': {
        'n_task':   16, 
        'env_fn':   place_expert,
        'eval_fn':  pick_place_eval,
        'agent-teacher': ('PandaPickPlaceDistractor', 'SawyerPickPlaceDistractor'),
        'render_hw': (200, 360), #(150, 270)
        },
    'stack': {
        'n_task':   6, 
        'env_fn':   stack_expert,
        'eval_fn':  block_stack_eval,
        'agent-teacher': ('PandaBlockStacking', 'SawyerBlockStacking'),
        'render_hw': (100, 180), ## older models used 100x200!!
        },
    'draw': {
        'n_task':   8,
        'env_fn':   draw_expert,
        'eval_fn':  draw_eval,
        'agent-teacher': ('PandaDrawer', 'SawyerDrawer'),
        'render_hw': (120, 180),
    },
    'press': {
        'n_task':   6,
        'env_fn':   press_expert,
        'eval_fn':  press_button_eval,
        'agent-teacher': ('PandaButton', 'SawyerButton'),
        'render_hw': (100, 180),
    },
    'open': {
        'n_task':   4,
        'env_fn':   door_expert,
        'eval_fn':  open_door_eval,
        'agent-teacher': ('PandaDoor', 'SawyerDoor'),
        'render_hw': (100, 180),
    },
}

def select_random_frames(frames, n_select, sample_sides=True):
    selected_frames = []
    clip = lambda x : int(max(0, min(x, len(frames) - 1)))
    per_bracket = max(len(frames) / n_select, 1)

    for i in range(n_select):
        n = clip(np.random.randint(int(i * per_bracket), int((i + 1) * per_bracket)))
        if sample_sides and i == n_select - 1:
            n = len(frames) - 1
        elif sample_sides and i == 0:
            n = 0
        selected_frames.append(n)

    if isinstance(frames, (list, tuple)):
        return [frames[i] for i in selected_frames]
    elif isinstance(frames, Trajectory):
        return [frames[i]['obs']['image'] for i in selected_frames]
        #return [frames[i]['obs']['image-state'] for i in selected_frames]
    return frames[selected_frames]
    
def build_env_context(img_formatter, T_context=4, ctr=0, env_name='nut', 
    heights=100, widths=200, size=False, shape=False, color=False, gpu_id=0, ):
    create_seed = random.Random(None)
    create_seed = create_seed.getrandbits(32)
    controller = load_controller_config(default_controller='IK_POSE')
    assert gpu_id != -1
    build_task = TASK_MAP.get(env_name, None)
    assert build_task, 'Got unsupported task '+env_name
    div = int(build_task['n_task'])
    env_fn = build_task['env_fn']
    agent_name, teacher_name = build_task['agent-teacher']

    task = ctr % div
    teacher_expert_rollout = env_fn( teacher_name, \
            controller_type=controller, task=task, \
            seed=create_seed, heights=heights, widths=widths, gpu_id=gpu_id)

    agent_env = env_fn( agent_name, \
            controller_type=controller, task=task, ret_env=True, seed=create_seed, 
             heights=heights, widths=widths, gpu_id=gpu_id)
    
    assert isinstance(teacher_expert_rollout, Trajectory)
    context = select_random_frames(teacher_expert_rollout, T_context, sample_sides=True)
    context = [img_formatter(i)[None] for i in context]
    # assert len(context ) == 6
    if isinstance(context[0], np.ndarray):
        context = torch.from_numpy(np.concatenate(context, 0))[None]
    else:
        context = torch.cat(context, dim=0)[None]


    return agent_env, context, task, teacher_expert_rollout
