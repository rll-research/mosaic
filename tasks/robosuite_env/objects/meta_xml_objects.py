import os
from robosuite.models.objects import MujocoObject
import copy
from copy import deepcopy
import xml.etree.ElementTree as ET

import robosuite.utils.macros as macros
from robosuite_env.objects.mujoco_xml import MujocoXML
from robosuite.utils.mjcf_utils import string_to_array, array_to_string, CustomMaterial, OBJECT_COLLISION_COLOR,\
                                       sort_elements, new_joint, add_prefix, add_material, find_elements



BASE_DIR = os.path.join(os.path.dirname(__file__), 'objects')

GEOMTYPE2GROUP = {
    "collision": {0},                 # If we want to use a geom for physics, but NOT visualize
    "visual": {1},                    # If we want to use a geom for visualization, but NOT physics
    "all": {0, 1},                    # If we want to use a geom for BOTH physics + visualization
}

GEOM_GROUPS = GEOMTYPE2GROUP.keys()

class MyMujocoXMLObject(MujocoObject, MujocoXML):
    """
    MujocoObjects that are loaded from xml files (by default, inherit all properties (e.g.: name)
    from MujocoObject class first!)
    Args:
        fname (str): XML File path
        name (str): Name of this MujocoXMLObject
        joints (None or str or list of dict): each dictionary corresponds to a joint that will be created for this
            object. The dictionary should specify the joint attributes (type, pos, etc.) according to the MuJoCo xml
            specification. If "default", a single free-joint will be automatically generated. If None, no joints will
            be created.
        obj_type (str): Geom elements to generate / extract for this object. Must be one of:
            :`'collision'`: Only collision geoms are returned (this corresponds to group 0 geoms)
            :`'visual'`: Only visual geoms are returned (this corresponds to group 1 geoms)
            :`'all'`: All geoms are returned
        duplicate_collision_geoms (bool): If set, will guarantee that each collision geom has a
            visual geom copy
    """

    def __init__(self, fname, name, joints="default", obj_type="all", duplicate_collision_geoms=True):
        MujocoXML.__init__(self, fname)
        # Set obj type and duplicate args
        assert obj_type in GEOM_GROUPS, "object type must be one in {}, got: {} instead.".format(GEOM_GROUPS, obj_type)
        self.obj_type = obj_type
        self.duplicate_collision_geoms = duplicate_collision_geoms

        # Set name
        self._name = name

        # joints for this object
        if joints == "default":
            self.joint_specs = [self.get_joint_attrib_template()]  # default free joint
        elif joints is None:
            self.joint_specs = []
        else:
            self.joint_specs = joints
        # Make sure all joints have names!
        for i, joint_spec in enumerate(self.joint_specs):
            if "name" not in joint_spec:
                joint_spec["name"] = "joint{}".format(i)

        # Lastly, parse XML tree appropriately
        self._obj = self._get_object_subtree()

        # Extract the appropriate private attributes for this
        self._get_object_properties()

    def _get_object_subtree(self):
        # Parse object
        obj = copy.deepcopy(self.worldbody.find("./body/body[@name='object']"))
        # Rename this top level object body (will have self.naming_prefix added later)
        obj.attrib["name"] = "main"
        # Get all geom_pairs in this tree
        geom_pairs = self._get_geoms(obj)

        # Define a temp function so we don't duplicate so much code
        obj_type = self.obj_type

        def _should_keep(el):
            return int(el.get("group")) in GEOMTYPE2GROUP[obj_type]

        # Loop through each of these pairs and modify them according to @elements arg
        for i, (parent, element) in enumerate(geom_pairs):
            # Delete non-relevant geoms and rename remaining ones
            if not _should_keep(element):
                parent.remove(element)
            else:
                g_name = element.get("name")
                g_name = g_name if g_name is not None else f"g{i}"
                element.set("name", g_name)
                # Also optionally duplicate collision geoms if requested (and this is a collision geom)
                if self.duplicate_collision_geoms and element.get("group") in {None, "0"}:
                    parent.append(self._duplicate_visual_from_collision(element))
                    # Also manually set the visual appearances to the original collision model
                    element.set("rgba", array_to_string(OBJECT_COLLISION_COLOR))
                    if element.get("material") is not None:
                        del element.attrib["material"]
        # add joint(s)
        for joint_spec in self.joint_specs:
            obj.append(new_joint(**joint_spec))
        # Lastly, add a site for this object
        template = self.get_site_attrib_template()
        template["rgba"] = "1 0 0 0"
        template["name"] = "default_site"
        obj.append(ET.Element("site", attrib=template))

        return obj

    def exclude_from_prefixing(self, inp):
        """
        By default, don't exclude any from being prefixed
        """
        return False

    def _get_object_properties(self):
        """
        Extends the base class method to also add prefixes to all bodies in this object
        """
        super()._get_object_properties()
        add_prefix(root=self.root, prefix=self.naming_prefix, exclude=self.exclude_from_prefixing)

    @staticmethod
    def _duplicate_visual_from_collision(element):
        """
        Helper function to duplicate a geom element to be a visual element. Namely, this corresponds to the
        following attribute requirements: group=1, conaffinity/contype=0, no mass, name appended with "_visual"
        Args:
            element (ET.Element): element to duplicate as a visual geom
        Returns:
            element (ET.Element): duplicated element
        """
        # Copy element
        vis_element = deepcopy(element)
        # Modify for visual-specific attributes (group=1, conaffinity/contype=0, no mass, update name)
        vis_element.set("group", "1")
        vis_element.set("conaffinity", "0")
        vis_element.set("contype", "0")
        vis_element.set("mass", "1e-8")
        vis_element.set("name", vis_element.get("name") + "_visual")
        return vis_element

    def _get_geoms(self, root, _parent=None):
        """
        Helper function to recursively search through element tree starting at @root and returns
        a list of (parent, child) tuples where the child is a geom element
        Args:
            root (ET.Element): Root of xml element tree to start recursively searching through
            _parent (ET.Element): Parent of the root element tree. Should not be used externally; only set
                during the recursive call
        Returns:
            list: array of (parent, child) tuples where the child element is a geom type
        """
        # Initialize return array
        geom_pairs = []
        # If the parent exists and this is a geom element, we add this current (parent, element) combo to the output
        if _parent is not None and root.tag == "geom":
            geom_pairs.append((_parent, root))
        # Loop through all children elements recursively and add to pairs
        for child in root:
            geom_pairs += self._get_geoms(child, _parent=root)
        # Return all found pairs
        return geom_pairs

    @property
    def bottom_offset(self):
        bottom_site = self.worldbody.find("./body/site[@name='{}bottom_site']".format(self.naming_prefix))
        return string_to_array(bottom_site.get("pos"))

    @property
    def top_offset(self):
        top_site = self.worldbody.find("./body/site[@name='{}top_site']".format(self.naming_prefix))
        return string_to_array(top_site.get("pos"))

    @property
    def horizontal_radius(self):
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='{}horizontal_radius_site']".format(self.naming_prefix)
        )
        return string_to_array(horizontal_radius_site.get("pos"))[0]


class Shelf(MyMujocoXMLObject):
    def __init__(self, name='shelf'):
        super().__init__(os.path.join(BASE_DIR, 'shelf.xml'), name=name)

class Drawer(MyMujocoXMLObject):
    def __init__(self, name='drawer'):
        super().__init__(os.path.join(BASE_DIR, 'drawer.xml'), name=name)

class CoffeeMachine(MyMujocoXMLObject):
    def __init__(self, name='coffeemachine'):
        super().__init__(os.path.join(BASE_DIR, 'coffeemachine.xml'), name=name)

class ButtonMachine(MyMujocoXMLObject):
    def __init__(self, name='coffeemachine'):
        super().__init__(os.path.join(BASE_DIR, 'buttonmachine.xml'), name=name)

class Mug(MyMujocoXMLObject):
    def __init__(self, name='mug'):
        super().__init__(os.path.join(BASE_DIR, 'mug.xml'),
                         name=name,
                         obj_type="all")

class Bin(MyMujocoXMLObject):
    def __init__(self, name='bin'):
        super().__init__(os.path.join(BASE_DIR, 'bins.xml'),
                         name=name)

class HammerBlock(MyMujocoXMLObject):
    def __init__(self, name='hammerblock'):
        super().__init__(os.path.join(BASE_DIR, 'hammerblock.xml'), name=name)

class Basketball(MyMujocoXMLObject):
    def __init__(self, name='basketball'):
        super().__init__(os.path.join(BASE_DIR, 'basketball.xml'), name=name)

class BasketballWhite(MyMujocoXMLObject):
    def __init__(self, name='basketball'):
        super().__init__(os.path.join(BASE_DIR, 'basketball_2.xml'), name=name)

class BasketballRed(MyMujocoXMLObject):
    def __init__(self, name='basketball'):
        super().__init__(os.path.join(BASE_DIR, 'basketball_3.xml'), name=name)

class Hoop(MyMujocoXMLObject):
    def __init__(self, name='Hoop'):
        super().__init__(os.path.join(BASE_DIR, 'basketballhoop.xml'), name=name, joints=None)
