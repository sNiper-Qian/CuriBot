import numpy as np

GRASP_CODE = np.ones(7)
RELEASE_CODE = np.zeros(7)

def sample_pose_on_sphere(distance=0.01):
    """
    Sample a random pose on a sphere.
    """
    # Randomly sample a point on the sphere around the object
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi/2)
    x = distance * np.sin(phi) * np.cos(theta)
    y = distance * np.sin(phi) * np.sin(theta)
    z = distance * np.cos(phi)

    # Orientation is currently always (1, 0, 0, 0)
    # TODO
    quat = [1, 0, 0, 0]

    # # Draw the xyz
    # point = [x, y, z]
    # self.addDebugPoint(point, color=[1, 0, 0], size=0.01)

    # Return the new pose with the same orientation as the object
    return np.array([x, y, z, quat[0], quat[1], quat[2], quat[3]])

class SimpleWaypointSampler:
    def __init__(self):
        pass

    def sample_waypoints(self, object_ids):
        rel_waypoints = []
        first_obj_id, second_obj_id = 0, 1
        # Approach first object
        sampled_pose = sample_pose_on_sphere(distance=0.1)
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), np.array(sampled_pose)]))
        # Grasp first object
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), GRASP_CODE]))
        # Approach second object
        sampled_pose = np.array([0, 0, 0, 1, 0, 0, 0]) # zero displacement
        rel_waypoints.append(np.concatenate([np.array(second_obj_id).reshape(-1,), np.array(sampled_pose)]))
        # Release first object
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), RELEASE_CODE]))
        rel_waypoints = np.stack(rel_waypoints)
        return rel_waypoints