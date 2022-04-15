from robosuite.utils.mjcf_utils import CustomMaterial

tex_attrib = {
            "type": "cube",
        }
mat_attrib = {
    "texrepeat": "1 1",
    "specular": "0.4",
    "shininess": "0.1",
}
redwood = CustomMaterial(
    texture="WoodRed",
    tex_name="redwood",
    mat_name="redwood_mat",
    tex_attrib=tex_attrib,
    mat_attrib=mat_attrib,
)
greenwood = CustomMaterial(
    texture="WoodGreen",
    tex_name="greenwood",
    mat_name="greenwood_mat",
    tex_attrib=tex_attrib,
    mat_attrib=mat_attrib,
)
bluewood = CustomMaterial(
    texture="WoodBlue",
    tex_name="bluewood",
    mat_name="bluewood_mat",
    tex_attrib=tex_attrib,
    mat_attrib=mat_attrib,
)

grayplaster = CustomMaterial(
    texture="PlasterGray",
    tex_name="grayplaster",
    mat_name="grayplaster_mat",
    tex_attrib=tex_attrib,
    mat_attrib=mat_attrib,
)

lemon = CustomMaterial(
    texture="Lemon",
    tex_name="lemon",
    mat_name="lemon_mat",
    tex_attrib=tex_attrib,
    mat_attrib=mat_attrib,
)

darkwood = CustomMaterial(
    texture="WoodDark",
    tex_name="darkwood",
    mat_name="darkwood_mat",
    tex_attrib=tex_attrib,
    mat_attrib=mat_attrib,
)