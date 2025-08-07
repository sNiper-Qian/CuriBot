import trimesh
import numpy as np
import os
import argparse
from trimesh.transformations import rotation_matrix
from scipy.spatial import cKDTree

def is_radially_symmetric(mesh: trimesh.Trimesh, tol: float = 1e-3) -> bool:
    """
    Check if the mesh approximates a perfect circle (disk) in some plane.

    It uses PCA to find the best-fit plane, then ensures all points lie close to that plane
    and have nearly constant distance from the centroid within that plane.

    Args:
        mesh (trimesh.Trimesh): The mesh to test.
        tol (float): Tolerance for planarity and radial uniformity.

    Returns:
        bool: True if approximates a circle within tolerance.
    """
    verts = mesh.vertices - mesh.centroid
    cov = np.cov(verts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Normal is eigenvector of smallest variance
    normal = eigvecs[:, 0]

    print(eigvals[0], tol**2)
    # Check planarity: small variance along normal
    if eigvals[0] > tol**2:
        print(f"Not planar: variance {eigvals[0]:.6f} exceeds tol^2 {tol**2:.6f}")
        return False

    # Project onto plane
    proj = verts - np.outer(verts.dot(normal), normal)
    dists = np.linalg.norm(proj, axis=1)
    radial_diff = dists.max() - dists.min()
    print(f"Radial spread: {radial_diff:.6f}")
    return radial_diff <= tol

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


def is_rotational_symmetric(mesh: trimesh.Trimesh,
                             axis: np.ndarray = np.array([0, 1, 0]),
                             tol: float = 0.08,
                             angles: list = None) -> bool:
    """
    Check if mesh is invariant under rotations about a given axis.

    Args:
        mesh (trimesh.Trimesh): The mesh to test.
        axis (np.ndarray): Unit vector direction of rotation axis.
        tol (float): Maximum allowed distance after rotation.
        angles (list): List of angles (radians) to test. If None, defaults to [pi, pi/2, pi/3, pi/4].

    Returns:
        bool: True if symmetric for all angles within tolerance.
    """
    if angles is None:
        angles = [np.pi, np.pi/2, np.pi/3, np.pi/4]

    verts = mesh.vertices.copy() - mesh.centroid
    tree = cKDTree(verts)

    for theta in angles:
        R = rotation_matrix(theta, axis[:3])[:3, :3]
        rotated = (verts @ R.T)
        dists, _ = tree.query(rotated, k=1)
        max_dist = dists.max()
        print(f"Rotation {theta:.2f} rad -> max distance: {max_dist:.6f}")
        print(max_dist, tol)
        if max_dist > tol:
            return False
    return True
    

def main():
    parser = argparse.ArgumentParser(
        description="Check mesh symmetry using PyBullet, Trimesh, and KD-tree methods."
    )
    parser.add_argument('obj_file', help='Path to the OBJ file to check symmetry')
    parser.add_argument('--mode', choices=['reflect', 'rotate', 'radial'], default='reflect',
                        help='Symmetry test: plane reflection, rotational, or radial (circle)')
    parser.add_argument('--axis', default='x',
                        help='Axis for reflection or rotation (x, y, z or list for rotate)')
    parser.add_argument('--tol', type=float, default=4e-1,
                        help='Tolerance for symmetry checks')
    args = parser.parse_args()
    object_paths = get_objects_paths(args.obj_file)[0]
    for obj_path in object_paths:
        mesh = trimesh.load(obj_path, force='mesh')
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Loaded geometry is not a single mesh.")

        result = is_rotational_symmetric(mesh)
        # result = is_radially_symmetric(mesh, tol=args.tol)

        print("Symmetric." if result else "Not symmetric.")
        #windowed size
        mesh.show(
            smooth=False,
            show_edges=True,
            show_faces=True,
            face_colors=[0.5, 0.5, 0.5, 1.0],
            edge_colors=[1.0, 0.0, 0.0, 1.0],
            window_size=(400, 300)
        )

if __name__ == '__main__':
    main()

# 02880940/