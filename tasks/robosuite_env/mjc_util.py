import robosuite
import robosuite_env
import os
import xml.etree.ElementTree as ET
import robosuite_env


def postprocess_model_xml(xml_str):
    """
    This function postprocesses the model.xml collected from a MuJoCo demonstration
    in order to make sure that the STL files can be found.
    """
    robosuite_path = os.path.split(robosuite.__file__)[0].split('/')
    robosuite_env_path =  os.path.split(robosuite_env.__file__)[0].split('/')

    # replace mesh and texture file paths
    tree = ET.fromstring(xml_str)
    root = tree
    asset = root.find("asset")
    meshes = asset.findall("mesh")
    textures = asset.findall("texture")
    all_elements = meshes + textures

    for elem in all_elements:
        old_path = elem.get("file")
        if old_path is None:
            continue
        old_path_split = old_path.split("/")

        if 'robosuite_env' in old_path:
            ind = [loc for loc, val in enumerate(old_path_split) if val == "robosuite_env"][0]
            path_split = robosuite_env_path
        else:
            ind = max(
                loc for loc, val in enumerate(old_path_split) if val == "robosuite"
            )  # last occurrence index
            path_split = robosuite_path
        new_path_split = path_split + old_path_split[ind + 1 :]
        new_path = "/".join(new_path_split)
        elem.set("file", new_path)

    return ET.tostring(root, encoding="utf8").decode("utf8")

def xml_path_completion(xml_path):
    """
    Takes in a local xml path and returns a full path.
        if @xml_path is absolute, do nothing
        if @xml_path is not absolute, load xml that is shipped by the package
    Args:
        xml_path (str): local xml path
    Returns:
        str: Full (absolute) xml path
    """
    if xml_path.startswith("/"):
        full_path = xml_path
    else:
        full_path = os.path.join(robosuite_env.arena.assets_root, xml_path)
    return full_path
