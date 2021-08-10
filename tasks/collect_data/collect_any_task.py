from robosuite import load_controller_config
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

from multiprocessing import Pool, cpu_count
import functools
import os
import pickle as pkl
import json
import random
import torch
from os.path import join 

# To be crystal clear: each constructed "Environment" is defined by both (task_name, robot_name), e.g. 'PandaBasketball'
# but each task_name may have differnt sub-task ids: e.g. Basketball-task_00 is throwing the white ball into hoop #1
TASK_ENV_MAP = {
    'open':     {
        'n_task': 4,
        'env_fn': door_expert,
        'panda':  'PandaDoor',
        'sawyer': 'SawyerDoor',
        },
    'bask_hard': {
        'n_task':   12, 
        'env_fn':   basketball_expert,
        'panda':    'PandaBasketball',
        'sawyer':   'SawyerBasketball',
        },
    
    'nut_hard':  {
        'n_task':   9, 
        'env_fn':   nut_expert,
        'panda':    'PandaNutAssemblyDistractor',
        'sawyer':   'SawyerNutAssemblyDistractor',
        },
    
    'place': {
        'n_task':   16, 
        'env_fn':   place_expert,
        'panda':    'PandaPickPlaceDistractor',
        'sawyer':   'SawyerPickPlaceDistractor',
        },

    'stack': {
        'n_task':   6, 
        'env_fn':   stack_expert,
        'panda':    'PandaBlockStacking',
        'sawyer':   'SawyerBlockStacking',
        },
    'color_stack': {
        'n_task':   6, 
        'env_fn':   stack_expert,
        'panda':    'PandaBlockStacking',
        'sawyer':   'SawyerBlockStacking',
        },
    'shape_stack': {
        'n_task':   6, 
        'env_fn':   stack_expert,
        'panda':    'PandaBlockStacking',
        'sawyer':   'SawyerBlockStacking',
        },
    'draw': {
        'n_task':   8, 
        'env_fn':   draw_expert,
        'panda':    'PandaDrawer',
        'sawyer':   'SawyerDrawer',
        },
    'press': {
        'n_task':   6, 
        'env_fn':   press_expert,
        'panda':    'PandaButton',
        'sawyer':   'SawyerButton',
        },
}
    
ROBOT_NAMES = ['panda', 'sawyer']

def save_rollout(N, env_type, env_func, save_dir, n_tasks, env_seed=False, camera_obs=True, seeds=None, n_per_group=1, depth=False, renderer=False, \
    heights=100, widths=200, gpu_count=1, color=False, shape=False):
    if isinstance(N, int):
        N = [N]
    for n in N:
        # NOTE(Mandi): removed the 'continue' part, always writes new data 
        task = int((n % (n_tasks * n_per_group)) // n_per_group)
        seed = None if seeds is None else seeds[n]
        env_seed = seeds[n - n % n_per_group] if seeds is not None and env_seed else None
        config = load_controller_config(default_controller='IK_POSE')
        gpu_id = int(n % gpu_count)
        if color or shape:
            assert 'BlockStacking' in env_type, env_type 
            traj = env_func(env_type, controller_type=config, renderer=renderer, camera_obs=camera_obs, task=task, \
            seed=seed, env_seed=env_seed, depth=depth, widths=widths, heights=heights, gpu_id=gpu_id, color=color,shape=shape)
        else:
            traj = env_func(env_type, controller_type=config, renderer=renderer, camera_obs=camera_obs, task=task, \
                seed=seed, env_seed=env_seed, depth=depth, widths=widths, heights=heights, gpu_id=gpu_id)
            if len(traj) < 5: # try again
                traj = env_func(env_type, controller_type=config, renderer=renderer, camera_obs=camera_obs, task=task, \
                seed=seed, env_seed=env_seed, depth=depth, widths=widths, heights=heights, gpu_id=gpu_id)
            
         # let's make a new folder structure for easier dataloader construct: 
        # env_type/task_id/traj_idx.pkl, where idxes start from 0 within each sub-task
        group_count = n // (n_tasks * n_per_group)
        traj_idx = n % n_per_group + n_per_group * group_count 
        
        save_path = os.path.join(save_dir, 'task_{:02d}'.format(task))
        os.makedirs(save_path, exist_ok=1)
        file_name = os.path.join(save_path, 'traj{:03d}.pkl'.format(traj_idx))
        pkl.dump({
            'traj': traj, 
            'len': len(traj),
            'env_type': env_type, 
            'task_id': task}, open(file_name, 'wb'))
        del traj


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', default='./', help='Folder to save rollouts')
    parser.add_argument('--num_workers', default=cpu_count(), type=int, help='Number of collection workers (default=n_cores)')
    parser.add_argument('--N', default=10, type=int, help="Number of trajectories to collect")
    parser.add_argument('--per_task_group', default=100, type=int, help="Number of trajectories of same task in row")
    # NOTE(Mandi): these are new:
    parser.add_argument('--task_name', '-tsk', default='nut', type=str, help="Environment name")
    parser.add_argument('--robot', '-ro', default='panda', type=str, help="Robot name") 
    parser.add_argument('--overwrite', action='store_true', help="Carefully overwrite stuff only when specified")
    
    parser.add_argument('--collect_cam', action='store_true', help="If flag then will collect camera observation")
    parser.add_argument('--renderer', action='store_true', help="If flag then will display rendering GUI")
    parser.add_argument('--random_seed', action='store_true', help="If flag then will collect data from random envs")
    parser.add_argument('--n_env', default=None, type=int, help="Number of environments to collect from")
    parser.add_argument('--n_tasks', default=12, type=int, help="Number of tasks in environment")
    parser.add_argument('--give_env_seed', action='store_true', help="Maintain seperate consistent environment sampling seed (for multi obj envs)")
    parser.add_argument('--depth', action='store_true', help="Use this flag to collect depth observations")
    parser.add_argument('--heights', default=200, type=int, help="Render image height")
    parser.add_argument('--widths', default=200, type=int, help="Render image width")
    # for blockstacking only:
    parser.add_argument('--color', action='store_true')
    parser.add_argument('--shape', action='store_true')

    args = parser.parse_args()
    assert args.num_workers > 0, "num_workers must be positive!"

    if args.random_seed:
        assert args.n_env is None
        seeds = [None for _ in range(args.N)]
    elif args.n_env:
        envs, rng = [263237945 + i for i in range(args.n_env)], random.Random(385008283)
        seeds = [int(rng.choice(envs)) for _ in range(args.N)]
    else:
        n_per_group = args.per_task_group
        seeds = [263237945 + int(n // (args.n_tasks * n_per_group)) * n_per_group + n % n_per_group for n in range(args.N)]

    # select proper names and functions
    assert (args.task_name in args.save_dir and args.robot in args.save_dir), args.save_dir 
    
    assert args.task_name in TASK_ENV_MAP.keys(), 'Got unsupported task. name {}'.format(args.task_name)
    print("Collecting {} trajs for {}, image height={}, width={}, using {} subtasks, collecting depth observation? {}".format(
        args.N, args.task_name, args.heights, args.widths, args.n_tasks, args.depth))
    print('Saving path: ', args.save_dir)
    

    assert args.robot in ROBOT_NAMES, 'Got unsupported robot name {}'.format(args.robot)
    specs = TASK_ENV_MAP.get(args.task_name)
    env_name = specs.get(args.robot, None)
    env_fn = specs.get('env_fn', None)
    assert env_name and env_fn, env_name+'is unsupported'
    print("Making environment {} for robot {}, using env builder {}".format(
        env_name, args.robot, env_fn))
    
    # handle path info
    if not os.path.exists(args.save_dir):
        assert args.overwrite, "Make sure don't overwrite existing data unintendedly."
        os.makedirs(args.save_dir, exist_ok=1)
    assert os.path.isdir(args.save_dir), "directory specified but is a file and not directory! " + args.save_dir
    os.makedirs(args.save_dir, exist_ok=1)

    json.dump(
        {
            'robot':    args.robot,
            'task':     args.task_name,
            'env_type': env_name,
            'n_tasks':  args.n_tasks,
            'heights':  args.heights,
            'widths':   args.widths,
            'task_group_size': args.per_task_group,
        }, 
        open( join(args.save_dir, 'info.json'), 'w'))
    
    
    count = torch.cuda.device_count()
    print( "Distributing work to %s GPUs"%count )
    if args.num_workers == 1:
        save_rollout(
            N=list(range(args.N)),
            env_type=env_name, env_func=env_fn, save_dir=args.save_dir, n_tasks=args.n_tasks, \
            env_seed=args.give_env_seed, camera_obs=args.collect_cam, seeds=seeds, n_per_group=args.per_task_group, \
            depth=args.depth, renderer=args.renderer, \
            heights=args.heights, widths=args.widths, gpu_count=count, \
            color=args.color, shape=args.shape)
    else:
        assert not args.renderer, "can't display rendering when using multiple workers"

        with Pool(args.num_workers) as p:
            f = functools.partial(
                save_rollout, 
                env_type=env_name, env_func=env_fn, save_dir=args.save_dir, n_tasks=args.n_tasks, \
                env_seed=args.give_env_seed, camera_obs=args.collect_cam, seeds=seeds, n_per_group=args.per_task_group, \
                depth=args.depth, renderer=args.renderer, \
                heights=args.heights, widths=args.widths, gpu_count=count, \
                color=args.color, shape=args.shape)

            p.map(f, range(args.N))
