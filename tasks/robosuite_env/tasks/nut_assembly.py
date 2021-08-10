import sys
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
import numpy as np
from robosuite.models.tasks import ManipulationTask
from robosuite_env.arena import PegsArena
from robosuite_env.objects.custom_xml_objects import RoundNut3Object, RoundNut2Object, RoundNutObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
import robosuite.utils.transform_utils as T
import random
from robosuite_env.sampler import BoundarySampler

class DefaultNutAssembly(SingleArmEnv):
    def __init__(
        self,
        robots,
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
        placement_initializer=None,
        single_object_mode=0,
        default_peg=None,
        nut_type=None,
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
    ):
        # task settings
        self.single_object_mode = single_object_mode
        self.nut_to_id = {"nut0": 0, "nut1": 1, "nut2": 2}
        if default_peg is None:
            self.peg_mapping = [0, 1, 2]
        else:
            self.peg_mapping = [default_peg, default_peg, default_peg]
        self.default_peg = default_peg
        if nut_type is not None:
            assert (
                    nut_type in self.nut_to_id.keys()
            ), "invalid @nut_type argument - choose one of {}".format(
                list(self.nut_to_id.keys())
            )
            self.nut_id = self.nut_to_id[nut_type]  # use for convenient indexing
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

    def reward(self, action=None):
        """
        Reward function for the task.
        Sparse un-normalized reward:
          - a discrete reward of 1.0 per nut if it is placed around its correct peg
        Un-normalized components if using reward shaping, where the maximum is returned if not solved:
          - Reaching: in [0, 0.1], proportional to the distance between the gripper and the closest nut
          - Grasping: in {0, 0.35}, nonzero if the gripper is grasping a nut
          - Lifting: in {0, [0.35, 0.5]}, nonzero only if nut is grasped; proportional to lifting height
          - Hovering: in {0, [0.5, 0.7]}, nonzero only if nut is lifted; proportional to distance from nut to peg
        Note that a successfully completed task (nut around peg) will return 1.0 per nut irregardless of whether the
        environment is using sparse or shaped rewards
        Note that the final reward is normalized and scaled by reward_scale / 2.0 (or 1.0 if only a single nut is
        being used) as well so that the max score is equal to reward_scale
        Args:
            action (np.array): [NOT USED]
        Returns:
            float: reward value
        """
        # compute sparse rewards
        self._check_success()
        reward = np.sum(self.objects_on_pegs)

        # add in shaped rewards
        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += max(staged_rewards)
        if self.reward_scale is not None:
            reward *= self.reward_scale
            if self.single_object_mode == 0:
                reward /= 2.0
        return reward

    def staged_rewards(self):
        """
        Calculates staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.
        Returns:
            4-tuple:
                - (float) reaching reward
                - (float) grasping reward
                - (float) lifting reward
                - (float) hovering reward
        """

        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        # filter out objects that are already on the correct pegs
        active_nuts = []
        for i, nut in enumerate(self.nuts):
            if self.objects_on_pegs[i]:
                continue
            active_nuts.append(nut)

        # reaching reward governed by distance to closest object
        r_reach = 0.
        if active_nuts:
            # reaching reward via minimum distance to the handles of the objects
            dists = [
                self._gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=active_nut.important_sites["handle"],
                    target_type="site",
                    return_distance=True,
                ) for active_nut in active_nuts
            ]
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        # grasping reward for touching any objects of interest
        r_grasp = int(self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=[g for active_nut in active_nuts for g in active_nut.contact_geoms])
        ) * grasp_mult

        # lifting reward for picking up an object
        r_lift = 0.
        table_pos = np.array(self.sim.data.body_xpos[self.table_body_id])
        if active_nuts and r_grasp > 0.:
            z_target = table_pos[2] + 0.2
            object_z_locs = self.sim.data.body_xpos[[self.obj_body_id[active_nut.name]
                                                     for active_nut in active_nuts]][:, 2]
            z_dists = np.maximum(z_target - object_z_locs, 0.)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (
                    lift_mult - grasp_mult
            )

        # hover reward for getting object above peg
        r_hover = 0.
        if active_nuts:
            r_hovers = np.zeros(len(active_nuts))
            peg_body_ids = [self.peg1_body_id, self.peg2_body_id, self.peg3_body_id]
            for i, nut in enumerate(active_nuts):
                valid_obj = False
                peg_pos = None
                for nut_name, idn in self.nut_to_id.items():
                    if nut_name in nut.name.lower():
                        peg_pos = np.array(self.sim.data.body_xpos[peg_body_ids[self.peg_mapping[idn]]])[:2]
                        valid_obj = True
                        break
                if not valid_obj:
                    raise Exception("Got invalid object to reach: {}".format(nut.name))
                ob_xy = self.sim.data.body_xpos[self.obj_body_id[nut.name]][:2]
                dist = np.linalg.norm(peg_pos - ob_xy)
                r_hovers[i] = r_lift + (1 - np.tanh(10.0 * dist)) * (
                        hover_mult - lift_mult
                )
            r_hover = np.max(r_hovers)

        return r_reach, r_grasp, r_lift, r_hover

    def on_peg(self, obj_pos, peg_id):

        if peg_id == 0:
            peg_pos = np.array(self.sim.data.body_xpos[self.peg1_body_id])
        elif peg_id == 1:
            peg_pos = np.array(self.sim.data.body_xpos[self.peg2_body_id])
        else:
            peg_pos = np.array(self.sim.data.body_xpos[self.peg3_body_id])
        res = False
        if (
                abs(obj_pos[0] - peg_pos[0]) < 0.03
                and abs(obj_pos[1] - peg_pos[1]) < 0.03
                and obj_pos[2] < self.table_offset[2] + 0.05
        ):
            res = True
        return res

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = PegsArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # define nuts
        self.nuts = []
        nut_names = ("nut0", "nut1", "nut2")


        for i, (nut_cls, nut_name) in enumerate(zip(
                (RoundNutObject, RoundNut2Object, RoundNut3Object),
                nut_names,
        )):
            nut = nut_cls(name=nut_name)
            self.nuts.append(nut)

        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.nuts,
        )

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        self.placement_initializer.append_sampler(
            BoundarySampler(
                name="NutSampler",
                mujoco_objects=self.nuts,
                x_range=[-0.13, -0.09],
                y_range=[-0.30, 0.30],
                rotation=[np.pi - 0.1, np.pi + 0.1],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.02,
                addtional_dist=0.04,
            )
        )

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

        self.table_body_id = self.sim.model.body_name2id("table")
        self.peg1_body_id = self.sim.model.body_name2id("peg1")
        self.peg2_body_id = self.sim.model.body_name2id("peg2")
        self.peg3_body_id = self.sim.model.body_name2id("peg3")

        for nut in self.nuts:
            self.obj_body_id[nut.name] = self.sim.model.body_name2id(nut.root_body)
            self.obj_geom_id[nut.name] = [self.sim.model.geom_name2id(g) for g in nut.contact_geoms]

        # information of objects
        self.object_site_ids = [self.sim.model.site_name2id(nut.important_sites["handle"]) for nut in self.nuts]

        # keep track of which objects are on their corresponding pegs
        self.objects_on_pegs = np.zeros(len(self.nuts))

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
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Move objects out of the scene depending on the mode
        nut_names = {nut.name for nut in self.nuts}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(nut_names))
        elif self.single_object_mode == 2:
            self.obj_to_use = self.nuts[self.nut_id].name
        if self.single_object_mode in {1, 2}:
            nut_names.remove(self.obj_to_use)
            self.clear_objects(list(nut_names))

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].
        Important keys:
            `'robot-state'`: contains robot-centric information.
            `'object-state'`: requires @self.use_object_obs to be True. Contains object-centric information.
            `'image'`: requires @self.use_camera_obs to be True. Contains a rendered frame from the simulation.
            `'depth'`: requires @self.use_camera_obs and @self.camera_depth to be True.
            Contains a rendered depth map from the simulation
        Returns:
            OrderedDict: Observations from the environment
        """
        di = super()._get_observation()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix
            pr = self.robots[0].robot_model.naming_prefix

            # remember the keys to collect into object info
            object_state_keys = []

            # for conversion to relative gripper frame
            gripper_pose = T.pose2mat((di[pr + "eef_pos"], di[pr + "eef_quat"]))
            world_pose_in_gripper = T.pose_inv(gripper_pose)

            for i, nut in enumerate(self.nuts):
                obj_str = nut.name
                obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj_str]])
                obj_quat = T.convert_quat(
                    self.sim.data.body_xquat[self.obj_body_id[obj_str]], to="xyzw"
                )
                di["{}_pos".format(obj_str)] = obj_pos
                di["{}_quat".format(obj_str)] = obj_quat

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

    def _check_success(self):
        """
        Check if all nuts have been successfully placed around their corresponding pegs.
        Returns:
            bool: True if all nuts are placed correctly
        """
        # remember objects that are on the correct pegs
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        for i, nut in enumerate(self.nuts):
            obj_str = nut.name
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            dist = np.linalg.norm(gripper_site_pos - obj_pos)
            r_reach = 1 - np.tanh(10.0 * dist)
            self.objects_on_pegs[i] = int(self.on_peg(obj_pos, self.peg_mapping[i]) and r_reach < 0.6)

        if self.single_object_mode > 0:
            return self.objects_on_pegs[self.nut_id] > 0  # need one object on peg

        # returns True if all objects are on correct pegs
        return np.sum(self.objects_on_pegs) == len(self.nuts)

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the closest nut.
        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the closest nut
        if vis_settings["grippers"]:
            # find closest object
            dists = [
                self._gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=nut.important_sites["handle"],
                    target_type="site",
                    return_distance=True,
                ) for nut in self.nuts
            ]
            closest_nut_id = np.argmin(dists)
            # Visualize the distance to this target
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.nuts[closest_nut_id].important_sites["handle"],
                target_type="site",
            )


class NutAssembly(DefaultNutAssembly):
    def __init__(self, robots, single_object_mode=0, default_peg=0, no_clear=False, **kwargs):
        self._no_clear = no_clear
        self._default_peg = default_peg
        super().__init__(robots=robots, single_object_mode=single_object_mode, default_peg=default_peg, initialization_noise=None, **kwargs)

    def clear_objects(self, obj):
        if self._no_clear:
            return
        super().clear_objects(obj)

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

        if self.single_object_mode == 2:
            di['target-object'] = self.nut_id
            di['target-peg'] = self._default_peg

        return di

    def initialize_time(self, control_freq):
        self.sim.model.vis.quality.offsamples = 8
        super().initialize_time(control_freq)


class SawyerNutAssemblyDistractor(NutAssembly):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, force_object=None, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        items = ['nut0', 'nut1', 'nut2']
        obj = np.random.choice(items) if force_object is None else force_object
        obj = items[obj] if isinstance(obj, int) else obj
        super().__init__(robots=['Sawyer'], single_object_mode=2, nut_type=obj, no_clear=True, **kwargs)

class PandaNutAssemblyDistractor(NutAssembly):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, force_object=None, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        items = ['nut0', 'nut1', 'nut2']
        obj = np.random.choice(items) if force_object is None else force_object
        obj = items[obj] if isinstance(obj, int) else obj
        super().__init__(robots=['Panda'], single_object_mode=2, nut_type=obj, no_clear=True, **kwargs)

if __name__ == '__main__':
    from robosuite.environments.manipulation.pick_place import PickPlace
    import robosuite
    from robosuite.controllers import load_controller_config

    controller = load_controller_config(default_controller="IK_POSE")
    env = SawyerNutAssemblyDistractor(hard_reset=True, has_renderer=True, controller_configs=controller,
                  has_offscreen_renderer=False, reward_shaping=False, use_camera_obs=False, camera_heights=320,
                  camera_widths=320, render_camera='agentview')
    for i in range(1000):
        if i % 200 == 0:
            env.reset()
        low, high = env.action_spec
        action = np.random.uniform(low=low, high=high)
        env.step(action)
        env.render()
