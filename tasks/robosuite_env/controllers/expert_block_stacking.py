"""
New 0419: updated the 'status' generation to use more descriptive words
"""
import sys
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
import numpy as np
from robosuite_env import get_env
from mosaic.datasets import Trajectory
import pybullet as p
from pyquaternion import Quaternion
import random
from robosuite_env.custom_ik_wrapper import normalize_action
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2axisangle
from robosuite.utils import RandomizationError
import torch
import os
import mujoco_py
# in case rebuild is needed to use GPU render: sudo mkdir -p /usr/lib/nvidia-000
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
# pip uninstall mujoco_py; pip install mujoco_py 

def _clip_delta(delta, max_step=0.015):
    norm_delta = np.linalg.norm(delta)

    if norm_delta < max_step:
        return delta
    return delta / norm_delta * max_step


class BlockStackingController:
    def __init__(self, env, ranges, tries=0):
        self._env = env
        self._g_tol = 5e-2 ** (tries + 1)
        self.reset()
        self.ranges = ranges

    def _calculate_quat(self, angle):
        if "Sawyer" in self._env.robot_names:
            new_rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            return Quaternion(matrix=self._base_rot.dot(new_rot))
        return self._base_quat

    def _get_target(self):
        return self._env.sim.data.body_xpos[self._env.cubeB_body_id] + [0, 0, 0.09]

    def reset(self):
        self._clearance = 0.05

        if "Sawyer" in self._env.robot_names:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.13
            self._final_thresh = 1e-2
            self._base_rot = np.array([[0, 1, 0.], [1, 0, 0.], [0., 0., -1.]])
            self._base_quat = Quaternion(matrix=self._base_rot)
        elif "Panda" in self._env.robot_names:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.13
            self._final_thresh = 1e-2
            self._base_rot = np.array([[1, 0, 0.], [0, -1, 0.], [0., 0., -1.]])
            self._base_quat = Quaternion(matrix=self._base_rot)
        else:
            raise NotImplementedError

        self._t = 0
        self._intermediate_reached = False
        self._hover_delta = 0.15

    def _object_in_hand(self, obs):
        if np.linalg.norm(obs['cubeA_pos'] - obs[self._obs_name]) < 0.018:
            return True
        elif self._env._check_grasp(gripper=self._env.robots[0].gripper, object_geoms=self._env.cubeA):
            return True
        return False

    def _check_grasp(self):
        if self._env._check_grasp(gripper=self._env.robots[0].gripper, object_geoms=self._env.cubeA):
            return True
        return False

    def _get_target_pose(self, delta_pos, base_pos, quat, max_step=None):
        if max_step is None:
            max_step = self._default_speed

        #delta_pos = _clip_delta(delta_pos, max_step)

        if self.ranges.shape[0] == 7:
            aa = np.concatenate(([quat.angle / np.pi], quat.axis))
            if aa[0] < 0:
                aa[0] += 1
        else:
            quat = np.array([quat.x, quat.y, quat.z, quat.w])
            aa = quat2axisangle(quat)
        return normalize_action(np.concatenate((delta_pos + base_pos, aa)), self.ranges)

    def act(self, obs):
        status = 'start'
        if self._t == 0:
            self._start_grasp = -1
            self._finish_grasp = False

            try:
                y = -(obs['cubeA_pos'][1] - obs[self._obs_name][1])
                x = obs['cubeA_pos'][0] - obs[self._obs_name][0]
            except:
                import pdb;
                pdb.set_trace()
            
            angle = np.arctan2(y, x)
            self._target_quat = self._calculate_quat(angle)

        if self._start_grasp < 0 and self._t < 15:
            if np.linalg.norm(obs['cubeA_pos'] - obs[self._obs_name] + [0, 0, self._hover_delta]) < self._g_tol or self._t == 14:
                self._start_grasp = self._t

            quat_t = Quaternion.slerp(self._base_quat, self._target_quat, min(1, float(self._t) / 5))
            eef_pose = self._get_target_pose(
                obs['cubeA_pos'] - obs[self._obs_name] + [0, 0, self._hover_delta],
                obs['eef_pos'], quat_t)
            action = np.concatenate((eef_pose, [-1]))
            status = 'prepare grasping {}'.format(self._env.cube_names['cubeA'])

        elif self._t < self._start_grasp + 35 and not self._finish_grasp:
            if not self._object_in_hand(obs):
                eef_pose = self._get_target_pose(
                    obs['cubeA_pos'] - obs[self._obs_name] - [0, 0, self._clearance],
                    obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [-1]))
                self.object_pos = obs['cubeA_pos']
                status = 'reaching {}'.format(self._env.cube_names['cubeA'])
            else:
                eef_pose = self._get_target_pose(self.object_pos - obs[self._obs_name] + [0, 0, self._hover_delta],
                                                 obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [1]))
                status = '{} is in hand'.format(self._env.cube_names['cubeA'])
                if self._check_grasp():
                    self._finish_grasp = True
                    status = 'finished grasping {}'.format(self._env.cube_names['cubeA'])

        elif np.linalg.norm(self._get_target() - obs[self._obs_name]) > self._final_thresh and self._object_in_hand(obs):
            target = self._get_target()
            eef_pose = self._get_target_pose(target - obs[self._obs_name], obs['eef_pos'], self._target_quat)
            action = np.concatenate((eef_pose, [1]))
            status = 'moving {} to {}'.format(
                self._env.cube_names['cubeA'], self._env.cube_names['cubeB'])
        else:
            eef_pose = self._get_target_pose(np.zeros(3), obs['eef_pos'], self._target_quat)
            action = np.concatenate((eef_pose, [-1]))
            status = 'stacking {} on {}'.format(
                self._env.cube_names['cubeA'], self._env.cube_names['cubeB'])
        self._t += 1
        
        return action, status

    def disconnect(self):
        p.disconnect()


def get_expert_trajectory(env_type, controller_type, renderer=False, camera_obs=True, task=None, ret_env=False,
                          seed=None, env_seed=None, depth=False, heights=100, widths=180, gpu_id=0, **kwargs):
    if heights > 320:
        print("Warning! Rendering big images.")
    assert 'gpu' in str(mujoco_py.cymj), 'Make sure to render with GPU.'
    # reassign the gpu id
    visible_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    gpu_id = int(visible_ids[gpu_id])

    seed = seed if seed is not None else random.getrandbits(32)
    env_seed = seed if env_seed is None else env_seed
    seed_offset = sum([int(a) for a in bytes(env_type, 'ascii')])
    np.random.seed(env_seed)
    if 'Sawyer' in env_type:
        action_ranges = np.array([[-0.3, 0.3], [-0.3, 0.3], [0.78, 1.2], [-5, 5], [-5, 5], [-5, 5]])
    else:
        action_ranges = np.array([[-0.3, 0.3], [-0.3, 0.3], [0.78, 1.2], [-1, 1], [-1, 1], [-1, 1], [-1, 1]])

    success, use_object = False, None
    if task is not None:
        assert 0 <= task <= 5, "task should be in [0, 5]"
        use_object = int(task)

    if ret_env:
        while True:
            try:
                env = get_env(env_type, force_object=use_object, controller_configs=controller_type,
                              has_renderer=renderer, has_offscreen_renderer=camera_obs,
                              reward_shaping=False, use_camera_obs=camera_obs,
                              camera_depths=depth, ranges=action_ranges, camera_names="agentview", 
                              camera_heights=heights, camera_widths=widths,render_gpu_device_id=gpu_id, **kwargs)
                break
            except RandomizationError:
                pass
        return env

    tries = 0
    while True:
        try:
            env = get_env(env_type, force_object=use_object, controller_configs=controller_type,
                          has_renderer=renderer, has_offscreen_renderer=camera_obs,
                          reward_shaping=False, use_camera_obs=camera_obs, camera_depths=depth, 
                          ranges=action_ranges, camera_names="agentview",
                          camera_heights=heights, camera_widths=widths, render_gpu_device_id=gpu_id, **kwargs)
            break
        except RandomizationError:
            pass
    while not success:
        controller = BlockStackingController(env.env, tries=tries, ranges=action_ranges)
        np.random.seed(seed + int(tries // 3) + seed_offset)
        while True:
            try:
                obs = env.reset()
                break
            except RandomizationError:
                pass
        mj_state = env.sim.get_state().flatten()
        sim_xml = env.model.get_xml()
        traj = Trajectory(sim_xml)

        env.reset_from_xml_string(sim_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(mj_state)
        env.sim.forward()
        traj.append(obs, raw_state=mj_state, info={'status':'start'})
        for t in range(int(env.horizon // 10)):
            action, status = controller.act(obs)
            obs, reward, done, info = env.step(action)
            assert 'status' not in info.keys(), "Don't overwrite information returned from environment. "
            info['status'] = status
            if renderer:
                env.render()
            mj_state = env.sim.get_state().flatten()
            traj.append(obs, reward, done, info, action, mj_state)

            if reward:
                success = True
                break
        tries += 1

    if renderer:
        env.close()
    del controller
    del env
    return traj


if __name__ == '__main__':
    config = load_controller_config(default_controller='IK_POSE')
    traj = get_expert_trajectory('PandaBlockStacking', config, renderer=True, camera_obs=False, task=2, size=False, shape=False, color=False,
                                 render_camera='agentview')
