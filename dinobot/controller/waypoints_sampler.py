import numpy as np

GRASP_CODE = np.ones(7)
RELEASE_CODE = np.zeros(7)

def sample_pose_on_sphere(distance=0.01, only_xy=False):
    """
    Sample a random pose on a sphere.
    """
    # Randomly sample a point on the sphere around the object
    theta = np.random.uniform(0, 2 * np.pi)
    if only_xy:
        # If only_xy is True, we limit the phi to a small range to keep the point close to the xy plane
        phi = np.random.uniform(np.pi/2-0.1, np.pi/2)
    else:
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

class PickAndPlaceWaypointSampler:
    def __init__(self):
        pass

    def sample_waypoints(self, object_ids):
        rel_waypoints = []
        first_obj_id, second_obj_id = 0, 1
        # Pre-grasp pose
        sampled_pose = sample_pose_on_sphere(distance=0.2)
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), np.array(sampled_pose)]))
        # Approach first object
        sampled_pose = sample_pose_on_sphere(distance=0.1)
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), np.array(sampled_pose)]))
        # Grasp first object
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), GRASP_CODE]))
        # Pre-place pose
        sampled_pose = sample_pose_on_sphere(distance=0.2)
        rel_waypoints.append(np.concatenate([np.array(second_obj_id).reshape(-1,), np.array(sampled_pose)]))
        # Approach second object
        sampled_pose = sample_pose_on_sphere(distance=0.1)
        rel_waypoints.append(np.concatenate([np.array(second_obj_id).reshape(-1,), np.array(sampled_pose)]))
        # Release first object
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), RELEASE_CODE]))
        rel_waypoints = np.stack(rel_waypoints)
        return rel_waypoints

class PushTaskWaypointSampler:
    """
    Generates waypoints to perform a linear push on a single object.
    Includes a random pre-push pose, a contact point on the object perimeter,
    and a post-push endpoint at a specified distance.
    """
    def __init__(self):
        pass

    def sample_waypoints(self, object_ids):
        """
        Generates waypoints for pushing the specified object.

        Args:
            object_id (int): Identifier for the object to push.
            object_position (np.ndarray): 3D position [x, y, z] of the object center.

        Returns:
            np.ndarray: An N×4 array of rows [object_id, x, y, z].
        """
        object_id = 0
        # Sample a random contact point
        sampled_contact_point = sample_pose_on_sphere(distance=0.05, only_xy=True)

        rel_direction = sampled_contact_point[:3]
        rel_direction /= np.linalg.norm(rel_direction)  # Normalize the direction vector

        # Pre-push: stand-off before contact
        pre_push = 0.1 * rel_direction
        pre_push = np.concatenate([pre_push, np.array([1, 0, 0, 0])])  # Keep orientation as (1, 0, 0, 0)

        # Post-push: endpoint after sliding the object along the push direction
        post_push = -0.2 * rel_direction
        post_push = np.concatenate([post_push, np.array([1, 0, 0, 0])])

        # Assemble and return waypoints
        rel_waypoints = []
        rel_waypoints.append(np.concatenate([np.array([-1]).reshape(-1,), GRASP_CODE]))
        # rel_waypoints.append(np.concatenate([np.array([object_id]), pre_push]))
        rel_waypoints.append(np.concatenate([np.array([object_id]), sampled_contact_point]))
        rel_waypoints.append(np.concatenate([np.array([object_id]).reshape(-1,), GRASP_CODE]))
        rel_waypoints.append(np.concatenate([np.array([object_id]), post_push]))

        return np.stack(rel_waypoints)
