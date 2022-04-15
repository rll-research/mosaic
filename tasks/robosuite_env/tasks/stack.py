from robosuite.environments.manipulation.stack import Stack as DefaultStack
import sys
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
import numpy as np
from robosuite_env.arena import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
import robosuite.utils.transform_utils as T
from robosuite_env.sampler import BoundarySampler
from robosuite_env.tasks import bluewood, greenwood, redwood, grayplaster, lemon, darkwood

NAMES = {'r': 'red block', 'g': 'green block', 'b': 'blue block'}

class Stack(DefaultStack):
    def __init__(self, task_id, robots, size=False, shape=False, color=False, **kwargs):
        self.task_id = task_id
        self.random_size = size
        self.random_shape = shape
        self.random_color = color
        super().__init__(robots=robots, initialization_noise=None, **kwargs)

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])
        material_dict = {'g': greenwood, 'r': redwood, 'b': bluewood}
        color_dict = {'g': [0, 1, 0, 1], 'r': [1, 0, 0, 1], 'b': [0, 0, 1, 1]}
        task_name = ['rgb', 'rbg', 'bgr', 'brg', 'grb', 'gbr']
        task = task_name[self.task_id]
        size_noise = 0

        if self.random_size:
            while np.abs(size_noise) < 0.0015:
                size_noise = np.random.uniform(-0.004, 0.004)
        if self.random_color:
            material_dict = {'g': grayplaster, 'r': lemon, 'b': darkwood}
            color_dict = {'g': [0, 0, 0, 1], 'r': [0, 0, 0, 1], 'b': [0, 0, 0, 1]}

        if not self.random_shape:
            self.cubeA = BoxObject(
                name="cubeA",
                size_min=[0.024 + size_noise, 0.024 + size_noise, 0.024 + size_noise],
                size_max=[0.024 + size_noise, 0.024 + size_noise, 0.024 + size_noise],
                rgba=color_dict[task[0]],
                material=material_dict[task[0]],
            )
            self.cubeB = BoxObject(
                name="cubeB",
                size_min=[0.028 + size_noise, 0.028 + size_noise, 0.028 + size_noise],
                size_max=[0.028 + size_noise, 0.028 + size_noise, 0.028 + size_noise],
                rgba=color_dict[task[1]],
                material=material_dict[task[1]],
            )
            self.cubeC = BoxObject(
                name="cubeC",
                size_min=[0.024 + size_noise, 0.024 + size_noise, 0.024 + size_noise],
                size_max=[0.024 + size_noise, 0.024 + size_noise, 0.024 + size_noise],
                rgba=color_dict[task[2]],
                material=material_dict[task[2]],
            )
        else:
            self.cubeA = CylinderObject(
                name="cubeA",
                size_min=[0.021 + size_noise, 0.021 + size_noise],
                size_max=[0.021 + size_noise, 0.021 + size_noise],
                rgba=color_dict[task[0]],
                material=material_dict[task[0]],
                friction=2,
            )
            self.cubeB = CylinderObject(
                name="cubeB",
                size_min=[0.024 + size_noise, 0.024 + size_noise],
                size_max=[0.024 + size_noise, 0.024 + size_noise],
                rgba=color_dict[task[1]],
                material=material_dict[task[1]],
                friction=2,
            )
            self.cubeC = CylinderObject(
                name="cubeC",
                size_min=[0.021 + size_noise, 0.021 + size_noise],
                size_max=[0.021 + size_noise, 0.021 + size_noise],
                rgba=color_dict[task[2]],
                material=material_dict[task[2]],
                friction=2,
            )

        self.cubes = [self.cubeA, self.cubeC, self.cubeB]
        self.cube_names = {'cubeA': NAMES[task[0]], 'cubeB': NAMES[task[1]], 'cubeC':  NAMES[task[2]]}
        # Create placement initializer
        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cubes,
        )

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # each object should just be sampled in the bounds of the bin (with some tolerance)
        self.placement_initializer.append_sampler(
            BoundarySampler(
                name="ObjectSampler",
                mujoco_objects=self.cubes[:2],
                x_range=[-0.20, -0.15],
                y_range=[-0.12, 0.12],
                rotation=[0, np.pi / 16],
                rotation_axis='z',
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.000,
                addtional_dist=0.008
            )
        )

        self.placement_initializer.append_sampler(
            BoundarySampler(
                name="TargetSampler",
                mujoco_objects=self.cubeB,
                x_range=[0.10, 0.15],
                y_range=[-0.08, 0.08],
                rotation=[0, np.pi / 16],
                rotation_axis='z',
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.000,
                addtional_dist=0.001
            )
        )

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()
        if self.use_camera_obs:
            cam_name = self.camera_names[0]
            #in_hand_cam_name = self.camera_names[1]
            di['image'] = di[cam_name + '_image'].copy()
            #di['hand_image'] = di[in_hand_cam_name + '_image'].copy()
            del di[cam_name + '_image']
            #del di[in_hand_cam_name + '_image']
            if self.camera_depths[0]:
                di['depth'] = di[cam_name + '_depth'].copy()
                di['depth'] = ((di['depth'] - 0.95) / 0.05 * 255).astype(np.uint8)

        return di

    def initialize_time(self, control_freq):
        self.sim.model.vis.quality.offsamples = 8
        super().initialize_time(control_freq)


class SawyerBlockStacking(Stack):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, force_object=None, **kwargs):
        obj = np.random.randint(6) if force_object is None else force_object
        super().__init__(task_id=obj, robots=['Sawyer'], **kwargs)

class PandaBlockStacking(Stack):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, force_object=None, **kwargs):
        obj = np.random.randint(6) if force_object is None else force_object
        super().__init__(task_id=obj, robots=['Panda'], **kwargs)



if __name__ == '__main__':
    from robosuite.environments.manipulation.pick_place import PickPlace
    import robosuite
    from robosuite.controllers import load_controller_config


    controller = load_controller_config(default_controller="IK_POSE")
    env = SawyerBlockStacking(has_renderer=True, controller_configs=controller,
                  has_offscreen_renderer=False, reward_shaping=False, use_camera_obs=False, camera_heights=320,
                  camera_widths=320)
    for i in range(1000):
        if i % 200 == 0:
            env.reset()
            print(env.task_id)
        low, high = env.action_spec
        action = np.random.uniform(low=low, high=high)
        env.step(action)
        env.render()
