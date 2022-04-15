from robosuite.environments.manipulation.pick_place import PickPlace as DefaultPickPlace
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
import sys
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
import numpy as np
from robosuite_env.objects.custom_xml_objects import *
from robosuite_env.arena import TableArena, BinsArena
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


class PickPlace(SingleArmEnv):
    def __init__(
            self,
            robots,
            default_bin=3,
            env_configuration="default",
            controller_configs=None,
            gripper_types="default",
            initialization_noise="default",
            table_full_size=(0.8, 0.8, 0.05),
            table_friction=(1, 0.005, 0.0001),
            use_camera_obs=True,
            use_object_obs=True,
            reward_scale=1.0,
            reward_shaping=False,
            single_object_mode=0,
            object_type=None,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=20,
            horizon=1000,
            ignore_done=False,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=256,
            camera_depths=False,
            no_clear=False,
    ):
        # task settings
        self.single_object_mode = single_object_mode
        self.object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
        self.obj_names = ["Milk", "Bread", "Cereal", "Can"]
        if object_type is not None:
            assert (
                    object_type in self.object_to_id.keys()
            ), "invalid @object_type argument - choose one of {}".format(
                list(self.object_to_id.keys())
            )
            self.object_id = self.object_to_id[
                object_type
            ]  # use for convenient indexing
        self.obj_to_use = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.82))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs
        self._no_clear = no_clear
        self._default_bin = default_bin
        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def not_in_bin(self, obj_pos, bin_id):

        bin_x_low = self.bin2_pos[0]
        bin_y_low = self.bin2_pos[1]
        if bin_id == 0 or bin_id == 2:
            bin_x_low -= self.bin_size[0] / 2
        if bin_id < 2:
            bin_y_low -= self.bin_size[1] / 2

        bin_x_high = bin_x_low + self.bin_size[0] / 2
        bin_y_high = bin_y_low + self.bin_size[1] / 2

        res = True
        if (
                bin_x_low < obj_pos[0] < bin_x_high
                and bin_y_low < obj_pos[1] < bin_y_high
                and self.bin2_pos[2] < obj_pos[2] < self.bin2_pos[2] + 0.1
        ):
            res = False
        return res

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()

        # Additional object references from this env
        self.obj_body_id = {}
        self.obj_geom_id = {}

        # object-specific ids
        for obj in (self.visual_objects + self.objects):
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
            self.obj_geom_id[obj.name] = [self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

        # keep track of which objects are in their corresponding bins
        self.objects_in_bins = np.zeros(len(self.objects))

        # target locations in bin for each object type
        self.target_bin_placements = np.zeros((len(self.objects), 3))
        for i, obj in enumerate(self.objects):
            bin_id = i
            bin_x_low = self.bin2_pos[0]
            bin_y_low = self.bin2_pos[1]
            if bin_id == 0 or bin_id == 2:
                bin_x_low -= self.bin_size[0] / 2.
            if bin_id < 2:
                bin_y_low -= self.bin_size[1] / 2.
            bin_x_low += self.bin_size[0] / 4.
            bin_y_low += self.bin_size[1] / 4.
            self.target_bin_placements[i, :] = [bin_x_low, bin_y_low, self.bin2_pos[2]]

        if self.single_object_mode == 2:
            self.target_bin_placements = self.target_bin_placements[self._bin_mappings]

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # Set the visual object body locations
                if "visual" in obj.name.lower():
                    self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_pos
                    self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat
                else:
                    # Set the collision object joints
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Move objects out of the scene depending on the mode
        obj_names = {obj.name for obj in self.objects}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(obj_names))
        elif self.single_object_mode == 2:
            self.obj_to_use = self.objects[self.object_id].name
        if self.single_object_mode in {1, 2}:
            obj_names.remove(self.obj_to_use)
            if not self._no_clear:
                self.clear_objects(list(obj_names))
        if self.single_object_mode == 2:
            self._bin_mappings = np.arange(4)
            self._bin_mappings[:] = self._default_bin

    def _check_success(self):
        """
        Check if all objects have been successfully placed in their corresponding bins.
        Returns:
            bool: True if all objects are placed correctly
        """
        # remember objects that are in the correct bins
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        for i, obj in enumerate(self.objects):
            obj_str = obj.name
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            dist = np.linalg.norm(gripper_site_pos - obj_pos)
            r_reach = 1 - np.tanh(10.0 * dist)
            self.objects_in_bins[i] = int((not self.not_in_bin(obj_pos, i)) and r_reach < 0.6)

        if self.single_object_mode == 2:
            obj_str = self.objects[self.object_id].name
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            return not self.not_in_bin(obj_pos, self._bin_mappings[self.object_id])

        # returns True if a single object is in the correct bin
        if self.single_object_mode == 1:
            return np.sum(self.objects_in_bins) > 0

        # returns True if all objects are in correct bins
        return np.sum(self.objects_in_bins) == len(self.objects)

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        self.placement_initializer.append_sampler(
            BoundarySampler(
                name="CollisionObjectSampler",
                mujoco_objects=self.objects,
                x_range=[-0.13, 0.20],
                y_range=[-0.30, 0.30],
                rotation=[0, np.pi/4],
                rotation_axis='z',
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.02,
                addtional_dist=0.01,
            )
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
        )
        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # store some arena attributes
        self.table_size = mujoco_arena.table_full_size

        self.objects = []
        self.visual_objects = []
        for vis_obj_cls, obj_name in zip(
                (MilkVisualObject, BreadVisualObject, CerealVisualObject, CanVisualObject),
                self.obj_names,
        ):
            vis_name = "Visual" + obj_name
            vis_obj = vis_obj_cls(name=vis_name)
            self.visual_objects.append(vis_obj)


        object_seq = (MilkObject, BreadObject, CerealObject, CanObject)

        object_seq = object_seq

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


    def reward(self, action=None):
        return float(self._check_success())

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

    def __init__(self, force_object=None, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        items = ['milk', 'bread', 'cereal', 'can']
        obj = np.random.choice(items) if force_object is None else force_object
        obj = items[obj] if isinstance(obj, int) else obj
        super().__init__(robots=['Sawyer'], single_object_mode=2, object_type=obj, no_clear=True, **kwargs)


class PandaPickPlaceDistractor(PickPlace):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, force_object=None, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        items = ['milk', 'bread', 'cereal', 'can']
        obj = np.random.choice(items) if force_object is None else force_object
        obj = items[obj] if isinstance(obj, int) else obj
        super().__init__(robots=['Panda'], single_object_mode=2, object_type=obj, no_clear=True,  **kwargs)



if __name__ == '__main__':
    from robosuite.controllers import load_controller_config

    controller = load_controller_config(default_controller="IK_POSE")
    env = SawyerPickPlaceDistractor(has_renderer=True, controller_configs=controller, has_offscreen_renderer=False,
                                    reward_shaping=False,
                                    use_camera_obs=False, camera_heights=320, camera_widths=320)
    env.reset()
    for i in range(10000):
        if i % 200 == 0:
            env.reset()
        low, high = env.action_spec
        action = np.random.uniform(low=low, high=high)
        env.step(action)
        env.render()