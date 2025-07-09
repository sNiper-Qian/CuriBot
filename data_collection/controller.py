import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def quaternion_slerp(start_ori, end_ori, t):
    """
    Performs spherical linear interpolation (SLERP) between two quaternions.

    :param start_ori: list or array-like, quaternion [x, y, z, w]
    :param end_ori: list or array-like, quaternion [x, y, z, w]
    :param t: float in [0, 1], interpolation parameter
    :return: interpolated quaternion [x, y, z, w]
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError("Interpolation parameter t must be between 0 and 1.")

    times = [0, 1]
    rotations = R.from_quat([start_ori, end_ori])
    slerp = Slerp(times, rotations)
    interpolated = slerp(t)
    return interpolated.as_quat()

def linear_interpolate_cartesian_pose(current_pose, desired_pose, max_step):
    """
    Interpolate a straight-line path in Cartesian space between a current pose and desired pose.
    
    :param current_pose:  The starting pose (geometry_msgs/Pose).
    :param desired_pose:  The goal pose (geometry_msgs/Pose).
    :param max_step:      The maximum step (in meters) between consecutive waypoints.
    :return: list of intermediate waypoints (Pose), including the start and final pose.
             Returns an empty list if interpolation fails or inputs are invalid.
    """
    # 1. Validate input
    if max_step <= 0.0:
        # We require a positive step size
        return []

    # 2. Extract positions as numpy arrays
    start_pos = np.array(current_pose[:3])
    end_pos   = np.array(desired_pose[:3])

    # 3. Extract orientations as [x, y, z, w]
    start_ori = np.array(current_pose[3:])
    end_ori   = np.array(desired_pose[3:])

    # Normalize just in case
    start_norm = np.linalg.norm(start_ori)
    end_norm   = np.linalg.norm(end_ori)
    if start_norm < 1e-12 or end_norm < 1e-12:
        # Invalid orientation (zero length)
        return []

    start_ori /= start_norm
    end_ori   /= end_norm

    # 4. Compute the distance in position space
    distance = np.linalg.norm(end_pos - start_pos)

    # If distance is very small, check orientation difference
    if distance < 1e-6:
        waypoints = []
        # Always push back current_pose
        waypoints.append(current_pose.copy())
        # Check if there's a meaningful orientation difference
        angle = _quaternion_angle_shortest_path(start_ori, end_ori)
        if angle > 1e-6:
            waypoints.append(desired_pose.copy())
        return waypoints

    # 5. Determine how many steps needed based on max_step
    #    e.g., distance=1.0 and max_step=0.05 => ~20 steps
    num_steps = int(math.ceil(distance / max_step))

    # 6. Determine how many steps we need for orientation
    #    e.g., limit orientation change to 10 degrees per step => (10 * pi/180)
    angle = _quaternion_angle_shortest_path(start_ori, end_ori)
    max_orient_step = 10.0 * math.pi / 180.0  # 10 degrees in radians
    orient_steps = 0
    if angle > 1e-6:
        orient_steps = int(math.ceil(angle / max_orient_step))

    # Overall steps is the max
    steps = max(num_steps, orient_steps, 1)

    # 7. Generate intermediate waypoints
    waypoints = []
    for i in range(steps + 1):
        t = float(i) / float(steps)

        # Linear interpolation for position
        interp_pos = start_pos + t * (end_pos - start_pos)

        # Spherical linear interpolation for orientation
        interp_ori = quaternion_slerp(start_ori, end_ori, t)

        # Build a Pose
        pose = np.concatenate((interp_pos, interp_ori))

        waypoints.append(pose)

    return waypoints


def _quaternion_angle_shortest_path(q1, q2):
    """
    Returns the angle (in radians) between two quaternions along the shortest path.
    Both quaternions should be normalized.
    """
    # Dot product
    dot = np.dot(q1, q2)
    # Numerical stability; dot should be in [-1, 1]
    dot = max(min(dot, 1.0), -1.0)
    # Angle is 2 * acos(|dot|)
    return 2.0 * math.acos(abs(dot))

if __name__ == "__main__":
    # Example usage
    current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # [x, y, z, qx, qy, qz, qw]
    desired_pose = np.array([1.0, 1.0, 1.0, 0.7071, 0.7071, 0.0, 0.0])  # [x, y, z, qx, qy, qz, qw]
    max_step = 0.5

    waypoints = linear_interpolate_cartesian_pose(current_pose, desired_pose, max_step)
    for wp in waypoints:
        print(wp)