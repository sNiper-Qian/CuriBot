import os
import h5py
import numpy as np
import json

def get_objects_paths(objects_path):
    """
    # Find all .urdf files in the objects directory and its subdirectories.
    """
    with_texture = {}
    urdf_files = []
    for root, dirs, files in os.walk(objects_path):
        for file in files:
            if file.endswith('.obj'):
                # Construct the full path to the URDF file and add it to the list.
                urdf_files.append(os.path.join(root, file))
                # Check if the file has a texture (whether it has an images folder in the upper directory)
                if os.path.exists(os.path.join("/".join(str(root).split(os.sep)[:-1]), 'images')):
                    with_texture[os.path.join(root, file)] = True
                else:
                    with_texture[os.path.join(root, file)] = False
    return urdf_files, with_texture

if __name__ == "__main__":
    objects_path = "../ShapeNetExtracted"
    urdf_files, with_texture = get_objects_paths(objects_path)
    filtered_urdf_files = []
    # Filter out objects without textures
    for urdf_file in urdf_files:
        if with_texture[urdf_file]:
            filtered_urdf_files.append(urdf_file)
    print(f"Filtered {len(filtered_urdf_files)} objects with textures out of {len(urdf_files)} total objects.")
    # Save the filtered list to a json file
    with open("filtered_ShapeNet.json", "w") as f:
        json.dump(filtered_urdf_files, f, indent=4)
    # Read the list from the json file
    with open("filtered_ShapeNet.json", "r") as f:
        filtered_urdf_files = json.load(f)