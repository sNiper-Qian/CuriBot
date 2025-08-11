import os
import sys

"""
Simple script to wrap an .obj file into an .urdf file.
"""


def split_filename(string):
    path_to_file = os.path.dirname(string)
    filename, extension = os.path.splitext(os.path.basename(string))
    return path_to_file, filename, extension


def check_input(file_name):
    # if len(sys.argv) == 1:
    #     print("Provide a <file.obj> as argument")
    #     sys.exit()
    # elif len(sys.argv) > 2:
    #     print("Too many arguments")
    #     sys.exit()
    _, _, ext = split_filename(file_name)
    if ext != ".obj":
        print("Incorrect extension (<{}> instead of <.obj>)".format(ext))
        sys.exit()
    if not os.path.exists(file_name):
        print("The file <{}> does not exist".format(file_name))
        sys.exit()


def generate_output_name(file_name):
    path_to_file, filename, extension = split_filename(file_name)
    if path_to_file == "":
        new_name = filename + ".urdf"
    else:
        new_name = path_to_file + "/" + filename + ".urdf"
    return new_name


def check_output_file(file_name):
    output_name = generate_output_name(file_name)
    if os.path.exists(output_name):
        print("Warning: <{}> already exists. Do you want to continue and overwrite it? [y, n] > ".format(output_name), end="")
        ans = input().lower()
        if ans not in ["y", "yes"]:
            sys.exit()


def write_urdf_text(file_name):
    output_name = generate_output_name(file_name)
    _, name, _ = split_filename(file_name)
    print("Creation of <{}>...".format(output_name), end="")
    with open(output_name, "w") as f:
        text = """<?xml version="1.0" ?>
<robot name="{}.urdf">
  <link name="baseLink">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
       <mass value="0.0"/>
       <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{}.obj" scale="1.0 1.0 1.0"/>
      </geometry>
      <material name="white">
       <color rgba="1 1 1 1"/>
     </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{}.obj" scale="1.0 1.0 1.0"/>
        <!-- You could also specify the collision (for the {}) with a "box" tag: -->
        <!-- <box size=".06 .06 .06"/> -->
      </geometry>
    </collision>
  </link>
</robot>
        """.format(name, name, name, name)
        f.write(text)
        print(" done")


if __name__ == "__main__":
    import json
    with open("filtered_ShapeNet.json", "r") as f:
      objects_paths = json.load(f)
    for obj_path in objects_paths:
        file_name = obj_path
        print(f"Processing {file_name}")
        # Check the input file
        check_input(file_name)
        # Generate the output name
        new_full_filename = generate_output_name(file_name)
        # Check if the output file already exists
        check_output_file(file_name)
        # Write the URDF text
        write_urdf_text(file_name)
    # check_input(file_name)
    # new_full_filename = generate_output_name(file_name)
    # check_output_file()
    # write_urdf_text()
