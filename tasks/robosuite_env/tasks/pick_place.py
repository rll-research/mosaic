from robosuite.environments.manipulation.pick_place import PickPlace as DefaultPickPlace
import sys
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
import numpy as np
from robosuite_env.objects.custom_xml_objects import *
from robosuite_env.arena import BinsArena
from robosuite.models.objects import (
    MilkVisualObject,
    BreadVisualObject,
    CerealVisualObject,
    CanVisualObject,
)
from robosuite.models.tasks import ManipulationTask
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
import robosuite.utils.transform_utils as T


class PickPlace(DefaultPickPlace):
    def __init__(self, robots, randomize_goal=False, single_object_mode=0, default_bin=3, no_clear=False, force_success=False,
                 use_novel_objects=False, num_objects=4, **kwargs):
        self._randomize_goal = randomize_goal
        self._no_clear = no_clear
        self._default_bin = default_bin
        self._force_success = force_success
        self._was_closed = False
        self._use_novel_objects = use_novel_objects
        self._num_objects = num_objects
        if randomize_goal:
            assert single_object_mode == 2, "only works with single_object_mode==2!"
        super().__init__(robots=robots, single_object_mode=single_object_mode, initialization_noise=None, **kwargs)

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # can sample anywhere in bin
        bin_x_half = self.model.mujoco_arena.table_full_size[0] / 2 - 0.04
        bin_y_half = self.model.mujoco_arena.table_full_size[1] / 2 - 0.05

        # each object should just be sampled in the bounds of the bin (with some tolerance)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CollisionObjectSampler",
                mujoco_objects=self.objects,
                x_range=[-bin_x_half, bin_x_half],
                y_range=[-bin_y_half, bin_y_half],
                rotation=[0, np.pi / 4],
                rotation_axis='z',
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.bin1_pos,
                z_offset=0.,
            )
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["bins"]
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = BinsArena(
            bin1_pos=self.bin1_pos,
            table_full_size=self.table_full_size,
            table_friction=self.table_friction
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # store some arena attributes
        self.bin_size = mujoco_arena.table_full_size

        self.objects = []
        self.visual_objects = []
        for vis_obj_cls, obj_name in zip(
                (MilkVisualObject, BreadVisualObject, CerealVisualObject, CanVisualObject),
                self.obj_names,
        ):
            vis_name = "Visual" + obj_name
            vis_obj = vis_obj_cls(name=vis_name)
            self.visual_objects.append(vis_obj)

        randomized_object_list = [[MilkObject, MilkObject2, MilkObject3], [BreadObject, BreadObject2, BreadObject3],
                                  [CerealObject, CerealObject2, CerealObject3], [CanObject, CanObject2, CanObject3]]

        if self._use_novel_objects:
            idx = np.random.randint(0, 3, 4)
            object_seq = (
            randomized_object_list[0][idx[0]], randomized_object_list[1][idx[1]], randomized_object_list[2][idx[2]],
            randomized_object_list[3][idx[3]])
        else:
            object_seq = (MilkObject, BreadObject, CerealObject, CanObject)

        object_seq = object_seq[:self._num_objects]

        for obj_cls, obj_name in zip(
                object_seq,
                self.obj_names,
        ):
            obj = obj_cls(name=obj_name)
            self.objects.append(obj)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.visual_objects + self.objects,
        )
        compiler = self.model.root.find('compiler')
        compiler.set('inertiafromgeom', 'auto')
        if compiler.attrib['inertiagrouprange'] == "0 0":
            compiler.attrib.pop('inertiagrouprange')

        # Generate placement initializer
        self._get_placement_initializer()

    def clear_objects(self, obj):
        if self._no_clear:
            return
        super().clear_objects(obj)

    def _get_reference(self):
        super()._get_reference()
        if self.single_object_mode == 2:
            self.target_bin_placements = self.target_bin_placements[self._bin_mappings]

    def _reset_internal(self):
        self._was_closed = False
        if self.single_object_mode == 2:
            # randomly target bins if in single_object_mode==2
            self._bin_mappings = np.arange(self._num_objects)
            if self._randomize_goal:
                np.random.shuffle(self._bin_mappings)
            else:
                self._bin_mappings[:] = self._default_bin
        super()._reset_internal()

    def reward(self, action=None):
        if self.single_object_mode == 2:
            return float(self._check_success())
        return super().reward(action)

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        if self.single_object_mode == 2:
            obj_str = self.objects[self.object_id].name
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            return not self.not_in_bin(obj_pos, self._bin_mappings[self.object_id])
        return super()._check_success()

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
            # in_hand_cam_name = self.camera_names[1]
            di['image'] = di[cam_name + '_image'].copy()
            # di['hand_image'] = di[in_hand_cam_name + '_image'].copy()
            del di[cam_name + '_image']
            # del di[in_hand_cam_name + '_image']
            if self.camera_depths[0]:
                di['depth'] = di[cam_name + '_depth'].copy()
                di['depth'] = ((di['depth'] - 0.95) / 0.05 * 255).astype(np.uint8)
        if self.single_object_mode == 2:
            di['target-box-id'] = self._bin_mappings[self.object_id]
            di['target-object'] = self.object_id

        # add observation for all objects
        pr = self.robots[0].robot_model.naming_prefix

        # remember the keys to collect into object info
        object_state_keys = []

        # for conversion to relative gripper frame
        gripper_pose = T.pose2mat((di[pr + "eef_pos"], di[pr + "eef_quat"]))
        world_pose_in_gripper = T.pose_inv(gripper_pose)

        for i, obj in enumerate(self.objects):
            obj_str = obj.name
            obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj_str]])
            obj_quat = T.convert_quat(
                self.sim.data.body_xquat[self.obj_body_id[obj_str]], to="xyzw"
            )
            di["{}_pos".format(obj_str)] = obj_pos
            di["{}_quat".format(obj_str)] = obj_quat

            # get relative pose of object in gripper frame
            object_pose = T.pose2mat((obj_pos, obj_quat))
            rel_pose = T.pose_in_A_to_pose_in_B(object_pose, world_pose_in_gripper)
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            di["{}_to_{}eef_pos".format(obj_str, pr)] = rel_pos
            di["{}_to_{}eef_quat".format(obj_str, pr)] = rel_quat

            object_state_keys.append("{}_pos".format(obj_str))
            object_state_keys.append("{}_quat".format(obj_str))
            object_state_keys.append("{}_to_{}eef_pos".format(obj_str, pr))
            object_state_keys.append("{}_to_{}eef_quat".format(obj_str, pr))

        di["object-state"] = np.concatenate([di[k] for k in object_state_keys])

        return di

    def initialize_time(self, control_freq):
        self.sim.model.vis.quality.offsamples = 8
        super().initialize_time(control_freq)


class SawyerPickPlaceDistractor(PickPlace):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, force_object=None, randomize_goal=True, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        items = ['milk', 'bread', 'cereal', 'can']
        obj = np.random.choice(items) if force_object is None else force_object
        obj = items[obj] if isinstance(obj, int) else obj
        super().__init__(robots=['Sawyer'], single_object_mode=2, object_type=obj, no_clear=True, randomize_goal=randomize_goal, **kwargs)


class PandaPickPlaceDistractor(PickPlace):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, force_object=None, randomize_goal=True, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        items = ['milk', 'bread', 'cereal', 'can']
        obj = np.random.choice(items) if force_object is None else force_object
        obj = items[obj] if isinstance(obj, int) else obj
        super().__init__(robots=['Panda'], single_object_mode=2, object_type=obj, no_clear=True, randomize_goal=randomize_goal, **kwargs)



if __name__ == '__main__':
    from robosuite.controllers import load_controller_config

    controller = load_controller_config(default_controller="IK_POSE")
    env = SawyerPickPlaceDistractor(has_renderer=True, controller_configs=controller, has_offscreen_renderer=False,
                                    reward_shaping=False,
                                    use_camera_obs=False, camera_heights=320, camera_widths=320, use_novel_objects=True)
    env.reset()
    for i in range(10000):
        if i % 200 == 0:
            env.reset()
        low, high = env.action_spec
        action = np.random.uniform(low=low, high=high)
        env.step(action)
        env.render()