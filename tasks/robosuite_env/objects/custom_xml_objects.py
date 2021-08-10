import os
import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite_env.objects.meta_xml_objects import MyMujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string, find_elements

BASE_DIR = os.path.join(os.path.dirname(__file__), 'xml')


class DoorObject(MyMujocoXMLObject):
    """
    Door with handle (used in Door)
    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """
    def __init__(self, name, friction=None, damping=None, lock=False):
        super().__init__(os.path.join(BASE_DIR, 'door.xml'),
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)

        # Set relevant body names
        self.door_body = self.naming_prefix  + "door"
        self.frame_body = self.naming_prefix +"frame"
        self.latch_body = self.naming_prefix  +"latch"
        self.hinge_joint = self.naming_prefix +"hinge"

        self.lock = lock
        self.friction = friction
        self.damping = damping
        if self.friction is not None:
            self._set_door_friction(self.friction)
        if self.damping is not None:
            self._set_door_damping(self.damping)

    def _set_door_friction(self, friction):
        """
        Helper function to override the door friction directly in the XML
        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):
        """
        Helper function to override the door friction directly in the XML
        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("damping", array_to_string(np.array([damping])))

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries
                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({
            "handle": self.naming_prefix + "handle",
            "handle_start": self.naming_prefix + "handle_start",
            "handle_end": self.naming_prefix + "handle_end"
        })
        return dic

class DoorObject2(MyMujocoXMLObject):
    """
    Door with handle (used in Door)
    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """
    def __init__(self, name, friction=None, damping=None, lock=False):
        super().__init__(os.path.join(BASE_DIR, 'door_2.xml'),
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)

        # Set relevant body names
        self.door_body = self.naming_prefix  + "door"
        self.frame_body = self.naming_prefix +"frame"
        self.latch_body = self.naming_prefix  +"latch"
        self.hinge_joint = self.naming_prefix +"hinge"

        self.lock = lock
        self.friction = friction
        self.damping = damping
        if self.friction is not None:
            self._set_door_friction(self.friction)
        if self.damping is not None:
            self._set_door_damping(self.damping)

    def _set_door_friction(self, friction):
        """
        Helper function to override the door friction directly in the XML
        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):
        """
        Helper function to override the door friction directly in the XML
        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("damping", array_to_string(np.array([damping])))

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries
                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({
            "handle": self.naming_prefix + "handle",
            "handle_start": self.naming_prefix + "handle_start",
            "handle_end": self.naming_prefix + "handle_end",
        })
        return dic

class SmallDoorObject(MyMujocoXMLObject):
    """
    Door with handle (used in Door)
    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """
    def __init__(self, name, friction=None, damping=None, lock=False):
        super().__init__(os.path.join(BASE_DIR, 'small_door.xml'),
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)

        # Set relevant body names
        self.door_body = self.naming_prefix  + "door"
        self.frame_body = self.naming_prefix +"frame"
        self.latch_body = self.naming_prefix  +"latch"
        self.hinge_joint = self.naming_prefix +"hinge"

        self.lock = lock
        self.friction = friction
        self.damping = damping
        if self.friction is not None:
            self._set_door_friction(self.friction)
        if self.damping is not None:
            self._set_door_damping(self.damping)

    def _set_door_friction(self, friction):
        """
        Helper function to override the door friction directly in the XML
        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):
        """
        Helper function to override the door friction directly in the XML
        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("damping", array_to_string(np.array([damping])))

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries
                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({
            "handle": self.naming_prefix + "handle",
            "handle_start": self.naming_prefix + "handle_start",
            "handle_end": self.naming_prefix + "handle_end"
        })
        return dic

class SmallDoorObject2(MyMujocoXMLObject):
    """
    Door with handle (used in Door)
    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """
    def __init__(self, name, friction=None, damping=None, lock=False):
        super().__init__(os.path.join(BASE_DIR, 'small_door_2.xml'),
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)

        # Set relevant body names
        self.door_body = self.naming_prefix  + "door"
        self.frame_body = self.naming_prefix +"frame"
        self.latch_body = self.naming_prefix  +"latch"
        self.hinge_joint = self.naming_prefix +"hinge"

        self.lock = lock
        self.friction = friction
        self.damping = damping
        if self.friction is not None:
            self._set_door_friction(self.friction)
        if self.damping is not None:
            self._set_door_damping(self.damping)

    def _set_door_friction(self, friction):
        """
        Helper function to override the door friction directly in the XML
        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):
        """
        Helper function to override the door friction directly in the XML
        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("damping", array_to_string(np.array([damping])))

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries
                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({
            "handle": self.naming_prefix + "handle",
            "handle_start": self.naming_prefix + "handle_start",
            "handle_end": self.naming_prefix + "handle_end",
        })
        return dic

class RoundNutObject(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(os.path.join(BASE_DIR, 'round_nut.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries
                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({
            "handle": self.naming_prefix + "handle_site"
        })
        return dic

class RoundNut2Object(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(os.path.join(BASE_DIR, 'round_nut_2.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries
                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({
            "handle": self.naming_prefix + "handle_site"
        })
        return dic

class RoundNut3Object(MujocoXMLObject):
    """
    Round nut (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(os.path.join(BASE_DIR, 'round_nut_3.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries
                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({
            "handle": self.naming_prefix + "handle_site"
        })
        return dic

class SquareNutObject(MujocoXMLObject):
    """
    Square nut object (used in NutAssembly)
    """

    def __init__(self, name):
        super().__init__(os.path.join(BASE_DIR, 'square_nut.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries
                :`'handle'`: Name of nut handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({
            "handle": self.naming_prefix + "handle_site"
        })
        return dic


class CerealObject(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name='cereal'):
        super().__init__(os.path.join(BASE_DIR, 'cereal.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class BreadObject(MujocoXMLObject):

    def __init__(self, name='bread'):
        super().__init__(os.path.join(BASE_DIR, 'bread.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class MilkObject(MujocoXMLObject):
    """
    Coke can object (used in SawyerPickPlace)
    """

    def __init__(self, name='milk'):
        super().__init__(os.path.join(BASE_DIR, 'milk.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class CokeCan(MujocoXMLObject):
    """
    Coke can object (used in SawyerPickPlace)
    """

    def __init__(self, name='can'):
        super().__init__(os.path.join(BASE_DIR, 'coke.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)
CanObject = CokeCan


class CerealObject2(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name='cereal'):
        super().__init__(os.path.join(BASE_DIR, 'cereal_2.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class BreadObject2(MujocoXMLObject):

    def __init__(self, name='bread'):
        super().__init__(os.path.join(BASE_DIR, 'bread_2.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class MilkObject2(MujocoXMLObject):
    """
    Coke can object (used in SawyerPickPlace)
    """

    def __init__(self, name='milk'):
        super().__init__(os.path.join(BASE_DIR, 'milk_2.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class CanObject2(MujocoXMLObject):
    def __init__(self, name='can'):
        super().__init__(os.path.join(BASE_DIR, 'coke_2.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class CerealObject3(MujocoXMLObject):
    """
    Cereal box object (used in PickPlace)
    """

    def __init__(self, name='cereal'):
        super().__init__(os.path.join(BASE_DIR, 'cereal_3.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class BreadObject3(MujocoXMLObject):

    def __init__(self, name='bread'):
        super().__init__(os.path.join(BASE_DIR, 'bread_3.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class MilkObject3(MujocoXMLObject):
    """
    Coke can object (used in SawyerPickPlace)
    """

    def __init__(self, name='milk'):
        super().__init__(os.path.join(BASE_DIR, 'milk_3.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class CanObject3(MujocoXMLObject):
    def __init__(self, name='can'):
        super().__init__(os.path.join(BASE_DIR, 'coke_3.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)



class SpriteCan(MujocoXMLObject):
    def __init__(self, name='sprite'):
        super().__init__(os.path.join(BASE_DIR, 'sprite.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class MDewCan(MujocoXMLObject):

    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'mdew.xml'), 'mdew')


class FantaCan(MujocoXMLObject):

    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'fanta.xml'), 'fanta')


class DrPepperCan(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'drpepper.xml'), 'drpepper')


class CheezeItsBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'cheezeits.xml'), 'cheezeits')


class FritoBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'frito.xml'), 'frito')


class BounceBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'bounce.xml'), 'bounce')


class CleanBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'clean.xml'), 'clean')


class WooliteBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'woolite.xml'), 'woolite')


class Lemon(MujocoXMLObject):
    def __init__(self, name='lemon'):
        super().__init__(os.path.join(BASE_DIR, 'lemon.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class Pear(MujocoXMLObject):
    def __init__(self, name='pear'):
        super().__init__(os.path.join(BASE_DIR, 'pear.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class Banana(MujocoXMLObject):
    def __init__(self, name='banana'):
        super().__init__(os.path.join(BASE_DIR, 'banana.xml'),
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class Orange(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'orange.xml'), 'orange')


class RedChips(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'redchips.xml'), 'redchips')


class PurpleChips(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'purplechips.xml'), 'purplechips')


class DutchChips(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'dutchchips.xml'), 'dutchchips')


class Whiteclaw(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'whiteclaw.xml'), 'whiteclaw')


class AltoidBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'altoid.xml'), 'altoid')


class CandyBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'candy.xml'), 'candy')


class CardboardBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'cardboard.xml'), 'cardboard')


class ClaratinBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'claratin.xml'), 'claratin')


class DoveBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'dove.xml'), 'dove')


class FiveBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'five.xml'), 'five')


class MotrinBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'motrin.xml'), 'motrin')


class TicTacBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'tictac.xml'), 'tictac')


class ZertecBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'zertec.xml'), 'zertec')

class PlateWithHoleObject(MujocoXMLObject):
    """
    Square plate with a hole in the center (used in PegInHole)
    """

    def __init__(self, name):
        super().__init__(os.path.join(BASE_DIR, 'peg_with_hole.xml'),
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)
