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


class BasketballController:
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


    def get_target_loc(self):
        names = ['place1_goal', 'place2_goal', 'place3_goal', 'place4_goal']
        target_loc = self._env.sim.data.site_xpos[self._env.sim.model.site_name2id(names[self._env.hoop_id])]
        return target_loc

    def reset(self):
        self._target_loc = self.get_target_loc() + [0, 0, 0.04]
        self._clearance = 0.05

        if "Sawyer" in self._env.robot_names:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.13
            self._final_thresh = 2e-2
            self._base_rot = np.array([[1, 0, 0.], [0, -1, 0.], [0., 0., -1.]])
            self._base_quat = Quaternion(matrix=self._base_rot)
        elif "Panda" in self._env.robot_names:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.13
            self._final_thresh = 3e-2
            self._base_rot = np.array([[1, 0., 0], [0, -1,  0.], [0., 0., -1.]])
            self._base_quat = Quaternion(matrix=self._base_rot)
        else:
            raise NotImplementedError

        self._t = 0
        self._intermediate_reached = False
        self._hover_delta = 0.15

    def _object_in_hand(self, obs):
        if np.linalg.norm(obs['target_obj_pos'] - obs[self._obs_name]) < 0.02:
            return True
        elif self._env._check_grasp(gripper=self._env.robots[0].gripper, object_geoms=[g for g in self._env.balls[self._env.ball_id].contact_geoms]):
            return True
        return False

    def _check_grasp(self):
        if self._env._check_grasp(gripper=self._env.robots[0].gripper,
                                  object_geoms=[g for g in self._env.balls[self._env.ball_id].contact_geoms]):
            return True
        return False

    def _get_target_pose(self, delta_pos, base_pos, quat, max_step=None):
        if max_step is None:
            max_step = self._default_speed

        delta_pos = _clip_delta(delta_pos, max_step)

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

            y = -(obs['target_obj_pos'][1] - obs[self._obs_name][1])
            x = obs['target_obj_pos'][0] - obs[self._obs_name][0]

            angle = np.arctan2(y, x)
            self._target_quat = self._calculate_quat(angle)

        if self._start_grasp < 0 and self._t < 15:
            if np.linalg.norm(obs['target_obj_pos'] - obs[self._obs_name] + [0, 0, self._hover_delta]) < self._g_tol or self._t == 14:
                self._start_grasp = self._t

            quat_t = Quaternion.slerp(self._base_quat, self._target_quat, min(1, float(self._t) / 5))
            eef_pose = self._get_target_pose(
                obs['target_obj_pos'] - obs[self._obs_name] + [0, 0, self._hover_delta],
                obs['eef_pos'], quat_t)
            action = np.concatenate((eef_pose, [-1]))
            status = 'prepare_grasp'

        elif self._t < self._start_grasp + 45 and not self._finish_grasp:
            if not self._object_in_hand(obs):
                eef_pose = self._get_target_pose(
                    obs['target_obj_pos'] - obs[self._obs_name] - [0, 0, self._clearance],
                    obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [-1]))
                self.object_pos = obs['target_obj_pos']
                status = 'reaching_ball'
            else:
                eef_pose = self._get_target_pose(self.object_pos - obs[self._obs_name] + [0, 0, self._hover_delta],
                                                 obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [1]))
                if self._check_grasp():
                    self._finish_grasp = True
                status = 'ball_in_hand'

        elif abs(obs['target_obj_pos'][2] - self._target_loc[2]) > 0.02 and self._object_in_hand(obs) and not self._intermediate_reached:
            target = obs['target_obj_pos'].copy()
            target[2] = self._target_loc[2]
            eef_pose = self._get_target_pose(target - obs['target_obj_pos'], obs['eef_pos'], self._target_quat)
            action = np.concatenate((eef_pose, [1]))
            status = 'moving'
        elif np.linalg.norm(
                self._target_loc[:2] - obs['target_obj_pos'][:2]) > self._final_thresh and self._object_in_hand(obs):
            self._intermediate_reached = True
            target = self._target_loc
            eef_pose = self._get_target_pose(target - obs['target_obj_pos'], obs['eef_pos'], self._target_quat)
            action = np.concatenate((eef_pose, [1]))
            status = 'throwing'
        else:
            eef_pose = self._get_target_pose(np.zeros(3), obs['eef_pos'], self._target_quat)
            action = np.concatenate((eef_pose, [-1]))
            status = 'threw'
        self._t += 1
        return action, status

    def disconnect(self):
        p.disconnect()


def get_expert_trajectory(env_type, controller_type, renderer=False, camera_obs=True, task=None, ret_env=False,
                          seed=None, env_seed=None, depth=False, widths=200, heights=200, gpu_id=0, **kwargs):
    # assert widths == 180, widths
    assert 'gpu' in str(mujoco_py.cymj), 'Make sure to render with GPU to make eval faster'
    # reassign the gpu id
    visible_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    gpu_id = int(visible_ids[gpu_id])

    seed = seed if seed is not None else random.getrandbits(32)
    env_seed = seed if env_seed is None else env_seed
    seed_offset = sum([int(a) for a in bytes(env_type, 'ascii')])

    if 'Sawyer' in env_type:
        action_ranges = np.array([[-0.4, 0.4], [-0.4, 0.4], [0.5, 1.5], [-5, 5], [-5, 5], [-5, 5]])
    else:
        action_ranges = np.array([[-0.4, 0.4], [-0.4, 0.4], [0.5, 1.5], [-1, 1], [-1, 1], [-1, 1], [-1, 1]])

    success = False
    if task is not None:
        assert 0 <= task <= 11, "task should be in [0, 12]"
        ball_id = int(task // 4)
        hoop_id = task % 4

    if ret_env:
        while True:
            try:
                env = get_env(env_type, ball_id=ball_id, hoop_id=hoop_id, controller_configs=controller_type,
                              has_renderer=renderer, has_offscreen_renderer=camera_obs,
                              reward_shaping=False, use_camera_obs=camera_obs, camera_heights=heights, camera_widths=widths,
                              camera_depths=depth, ranges=action_ranges, camera_names="agentview", render_gpu_device_id=gpu_id,
                              **kwargs)
                break
            except RandomizationError:
                pass
        return env

    tries = 0
    while True:
        try:
            env = get_env(env_type, ball_id=ball_id, hoop_id=hoop_id, controller_configs=controller_type,
                          has_renderer=renderer, has_offscreen_renderer=camera_obs,
                          reward_shaping=False, use_camera_obs=camera_obs, camera_heights=heights,
                          camera_widths=widths, camera_depths=depth, ranges=action_ranges, render_gpu_device_id=gpu_id,
                          camera_names="agentview", **kwargs)
            break
        except RandomizationError:
            pass
    while not success:
        controller = BasketballController(env.env, tries=tries, ranges=action_ranges)
        np.random.seed(seed + int(tries) + seed_offset)
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
        traj.append(obs, raw_state=mj_state, info={'status': 'start'})
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
    traj = get_expert_trajectory('PandaBasketball', config, renderer=True, camera_obs=False, task=10, render_camera='agentview')
