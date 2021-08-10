from robosuite.wrappers.wrapper import Wrapper
import numpy as np
from pyquaternion import Quaternion
import robosuite.utils.transform_utils as T


def normalize_action(action, ranges):
    # normalizing action: pos + axis-angle
    norm_action = action.copy()
    for d in range(ranges.shape[0]):
        norm_action[d] = 2 * (norm_action[d] - ranges[d,0]) / (ranges[d,1] - ranges[d,0]) - 1
    return (norm_action * 128).astype(np.int32).astype(np.float32) / 128


def denormalize_action(norm_action, base_pos, base_quat, ranges):
    action = np.clip(norm_action.copy(), -1, 1)
    for d in range(ranges.shape[0]):
        action[d] = 0.5 * (action[d] + 1) * (ranges[d,1] - ranges[d,0]) + ranges[d,0]
    if norm_action.shape[0] == 7:
        cmd_quat = T.axisangle2quat(action[3:6])
        quat = T.quat_multiply(T.quat_inverse(base_quat), cmd_quat)
        aa = T.quat2axisangle(quat)
        return np.concatenate((action[:3] - base_pos, aa, action[6:]))
    else:
        cmd_quat = Quaternion(angle=action[3] * np.pi, axis=action[4:7])
        cmd_quat = np.array([cmd_quat.x, cmd_quat.y, cmd_quat.z, cmd_quat.w])
        quat = T.quat_multiply(T.quat_inverse(base_quat), cmd_quat)
        aa = T.quat2axisangle(quat)
        return np.concatenate((action[:3] - base_pos, aa, action[7:]))

def get_rel_action(action, base_pos, base_quat):
    if action.shape[0] == 7:
        cmd_quat = T.axisangle2quat(action[3:6])
        quat = T.quat_multiply(T.quat_inverse(base_quat), cmd_quat)
        aa = T.quat2axisangle(quat)
        return np.concatenate((action[:3] - base_pos, aa, action[6:]))
    else:
        cmd_quat = Quaternion(angle=action[3] * np.pi, axis=action[4:7])
        cmd_quat = np.array([cmd_quat.x, cmd_quat.y, cmd_quat.z, cmd_quat.w])
        quat = T.quat_multiply(T.quat_inverse(base_quat), cmd_quat)
        aa = T.quat2axisangle(quat)
        return np.concatenate((action[:3] - base_pos, aa, action[7:]))


def project_point(point, sim, camera='agentview', frame_width=320, frame_height=320):
    model_matrix = np.zeros((3, 4))
    model_matrix[:3, :3] = sim.data.get_camera_xmat(camera).T

    fovy = sim.model.cam_fovy[sim.model.camera_name2id(camera)]
    f = 0.5 * frame_height / np.tan(fovy * np.pi / 360)
    camera_matrix = np.array(((f, 0, frame_width / 2), (0, f, frame_height / 2), (0, 0, 1)))

    MVP_matrix = camera_matrix.dot(model_matrix)
    cam_coord = np.ones((4, 1))
    cam_coord[:3, 0] = point - sim.data.get_camera_xpos(camera)

    clip = MVP_matrix.dot(cam_coord)
    row, col = clip[:2].reshape(-1) / clip[2]
    row, col = row, frame_height - col
    return int(max(col, 0)), int(max(row, 0))


def post_proc_obs(obs, env):
    new_obs = {}
    from PIL import Image
    robot_name = env.robots[0].robot_model.naming_prefix
    for k in obs.keys():
        if k.startswith(robot_name):
            name = k[len(robot_name):]
            if isinstance(obs[k], np.ndarray):
                new_obs[name] = obs[k].copy()
            else:
                new_obs[name] = obs[k]
        else:
            if isinstance(obs[k], np.ndarray):
                new_obs[k] = obs[k].copy()
            else:
                new_obs[k] = obs[k]

    frame_height, frame_width = 320, 320
    if 'image' in obs:
        new_obs['image'] = obs['image'].copy()[::-1]
        frame_height, frame_width = new_obs['image'].shape[0], new_obs['image'].shape[1]
    if 'depth' in obs:
        new_obs['depth'] = obs['depth'].copy()[::-1]
    if 'hand_image' in obs:
        new_obs['hand_image'] = obs['hand_image'].copy()[::-1]

    aa = T.quat2axisangle(obs[robot_name+'eef_quat'])
    flip_points = np.array(project_point(obs[robot_name+'eef_pos'], env.sim, \
        frame_width=frame_width, frame_height=frame_height))
    flip_points[0] = frame_height - flip_points[0]
    flip_points[1] = frame_width - flip_points[1]
    new_obs['eef_point'] = flip_points
    new_obs['ee_aa'] = np.concatenate((obs[robot_name+'eef_pos'], aa)).astype(np.float32)
    return new_obs


class CustomIKWrapper(Wrapper):
    def __init__(self, env, ranges):
        super().__init__(env)
        self.action_repeat=5
        self.ranges = ranges

    def step(self, action):
        reward = -100.0
        for _ in range(self.action_repeat):
            base_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
            base_quat = self.env._eef_xquat
            rel_action = denormalize_action(action, base_pos, base_quat, self.ranges)
            obs, reward_t, done, info = self.env.step(rel_action)
            reward = max(reward, reward_t)
        return post_proc_obs(obs, self.env), reward, done, info

    def reset(self):
        obs = super().reset()
        return post_proc_obs(obs, self.env)

    def _get_observation(self):
        return post_proc_obs(self.env._get_observation(), self.env)
