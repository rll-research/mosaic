import cv2
import copy
import numpy as np

try:
    from mujoco_py import load_model_from_xml, MjSim, MjRenderContextOffscreen
    from tasks.robosuite_env import postprocess_model_xml
except:
    # in case experiments don't require rendering ignore failure
    pass


def _compress_obs(obs):
    if 'image' in obs:
        okay, im_string = cv2.imencode('.jpg', obs['image'])
        assert okay, "image encoding failed!"
        obs['image'] = im_string
    if 'depth' in obs:
        assert len(obs['depth'].shape) == 2 and obs['depth'].dtype == np.uint8, "assumes uint8 greyscale depth image!"
        depth_im = np.tile(obs['depth'][:,:,None], (1, 1, 3))
        okay, depth_string = cv2.imencode('.jpg', depth_im)
        assert okay, "depth encoding failed!"
        obs['depth'] = depth_string
    return obs


def _decompress_obs(obs):
    if 'image' in obs:
        obs['image'] = cv2.imdecode(obs['image'], cv2.IMREAD_COLOR)
    if 'depth' in obs:
        obs['depth'] = cv2.imdecode(obs['depth'], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    return obs


class Trajectory:
    def __init__(self, config_str=None):
        self._data = []
        self._raw_state = []
        self.set_config_str(config_str)

    def append(self, obs, reward=None, done=None, info=None, action=None, raw_state=None):
        """
        Logs observation and rewards taken by environment as well as action taken
        """
        obs, reward, done, info, action, raw_state = [copy.deepcopy(x) for x in [obs, reward, done, info, action, raw_state]]

        obs = _compress_obs(obs)
        self._data.append((obs, reward, done, info, action))
        self._raw_state.append(raw_state)

    @property
    def T(self):
        """
        Returns number of states
        """
        return len(self._data)

    def __getitem__(self, t):
        return self.get(t)

    def get(self, t, decompress=True):
        assert 0 <= t < self.T or -self.T < t <= 0, "index should be in (-T, T)"

        obs_t, reward_t, done_t, info_t, action_t = copy.deepcopy(self._data[t])
        if decompress:
            obs_t = _decompress_obs(obs_t)
        ret_dict = dict(obs=obs_t, reward=reward_t, done=done_t, info=info_t, action=action_t)

        for k in list(ret_dict.keys()):
            if ret_dict[k] is None:
                ret_dict.pop(k)
        return ret_dict

    def change_obs(self, t, obs):
        obs_t, reward_t, done_t, info_t, action_t = self._data[t]
        self._data[t] = obs, reward_t, done_t, info_t, action_t

    def __len__(self):
        return self.T

    def __iter__(self):
        for d in range(self.T):
            yield self.get(d)

    def get_raw_state(self, t):
        assert 0 <= t < self.T or -self.T < t <= 0, "index should be in (-T, T)"
        return copy.deepcopy(self._raw_state[t])

    def set_config_str(self, config_str):
        self._config_str = config_str

    @property
    def config_str(self):
        return self._config_str

class _HDF5BackedData:
    def __init__(self, hf, length):
        self._hf = hf
        self._len = length
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, index):
        group_t = self._hf[str(index)]
        obs_t = {}
        for k in group_t['obs'].keys():
            obs_t[k] = group_t['obs'][k][:]
        reward_t = group_t.get('reward', None)
        done_t = group_t.get('done', None)
        info_t = group_t.get('info', None)
        action_t = None
        if 'action' in group_t:
            action_t=group_t.get('action')[:]
        return obs_t, reward_t, done_t, info_t, action_t


class HDF5Trajectory(Trajectory):
    def __init__(self, fname=None, traj=None, config_str=None):
        self._hf_name = None
        if traj is not None:
            assert fname is None
            self.set_config_str(traj.config_str)
            self._data = copy.deepcopy(traj._data)
            self._raw_state = copy.deepcopy(traj._raw_state)
        elif fname is not None:
            self.load(fname)
        else:
            super().__init__(config_str)

    def append(self, obs, reward=None, done=None, info=None, action=None, raw_state=None):
        raise NotImplementedError("Cannot Append to HDF5 backed trajectory!")

    def load(self, fname):
        hf = h5py.File(fname, 'r')
        self._config_str = hf.get('config_str', None)
        self._raw_state = hf.get('raw_state', None)

        cntr = 0
        while str(cntr) in hf:
            cntr += 1
        self._data = _HDF5BackedData(hf, cntr)
        if self._raw_state is None:
            self._raw_state = [None for _ in range(cntr)]
    
    def to_pkl_traj(self):
        traj = Trajectory()
        traj._config_str = copy.deepcopy(self._config_str)
        traj._raw_state = copy.deepcopy(self._raw_state)
        traj._data = [self._data[t] for t in range(len(self._data))]
        return traj

    def save(self, fname):
        with h5py.File(fname, 'w') as hf:
            if self._config_str:
                hf.create_dataset('config_str', data=self._config_str)
            if any(self._raw_state):
                hf.create_dataset('raw_state', data=self._raw_state)

            cntr = 0
            for obs_t, reward_t, done_t, info_t, action_t in self._data:
                group_t = hf.create_group(str(cntr))
                if obs_t:
                    obs_group = group_t.create_group('obs')
                    for k, v in obs_t.items():
                        obs_group.create_dataset(k, data=v)
                if reward_t is not None:
                    group_t.create_dataset('reward', data=reward_t)
                if done_t is not None:
                    group_t.create_dataset('done', data=done_t)
                if info_t is not None:
                    group_t.create_dataset('info', data=info_t)
                if action_t is not None:
                    group_t.create_dataset('action', data=action_t)
                cntr += 1

class ImageRenderWrapper:
    def __init__(self, traj, height=320, width=320, depth=False, no_render=False):
        self._height = height
        self._width = width
        self._sim = None
        self._traj = traj
        self._depth = depth
        self._no_render = no_render

    def get(self, t, decompress=True):
        ret = self._traj[t]
        if decompress and 'image' not in ret['obs'] and not self._no_render:
            sim = self._get_sim()
            sim.set_state_from_flattened(self._traj.get_raw_state(t))
            sim.forward()
            if self._depth:
                image, depth = sim.render(camera_name='frontview', width=self._width, height=self._height, depth=True)
                ret['obs']['image'] = image[:,::-1]
                ret['obs']['depth'] = self._proc_depth(depth[:,::-1])
            else:
                ret['obs']['image'] = sim.render(camera_name='frontview', width=self._width, height=self._height, depth=False)[80:,::-1]
        return ret

    def _proc_depth(self, depth):
        if self._depth_norm == 'sawyer':
            return (depth - 0.992) / 0.0072
        return depth
    
    def _get_sim(self):
        if self._sim is not None:
            return self._sim

        xml = postprocess_model_xml(self._traj.config_str)
        self._depth_norm = None
        if 'sawyer' in xml:
            from hem.datasets.precompiled_models.sawyer import models
            self._sim = models[0]
            self._depth_norm = 'sawyer'
        elif 'baxter' in xml:
            from hem.datasets.precompiled_models.baxter import models
            self._sim = models[0]
        elif 'panda' in xml:
            from hem.datasets.precompiled_models.panda import models
            self._sim = models[0]
        else:
            model = load_model_from_xml(xml)
            model.vis.quality.offsamples = 8
            sim = MjSim(load_model_from_xml(xml))
            render_context = MjRenderContextOffscreen(sim)
            render_context.vopt.geomgroup[0] = 0
            render_context.vopt.geomgroup[1] = 1 
            sim.add_render_context(render_context)
            self._sim = sim

        return self._sim

    def __getitem__(self, t):
        return self.get(t)

    def __len__(self):
        return len(self._traj)
    
    def __iter__(self):
        for d in range(len(self._traj)):
            yield self.get(d)

    @property
    def config_str(self):
        return self._traj.config_str
