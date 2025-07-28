import numpy as np
from scipy.spatial.transform import Rotation as R

GRASP_CODE = np.ones(7)
RELEASE_CODE = np.zeros(7)
TASK_TYPES = ['pick_and_place', 'move_to_target', 'push', 'pull', 'pick']

def euler_to_quaternion(x, y, z):
    """
    Convert X-Y-Z Euler angles into quaternion (x, y, z, w).
    Angles are in radians.
    """
    cy = np.cos(z * 0.5)
    sy = np.sin(z * 0.5)
    cp = np.cos(y * 0.5)
    sp = np.sin(y * 0.5)
    cr = np.cos(x * 0.5)
    sr = np.sin(x * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = cr * cp * sy - sr * sp * cy
    y = cr * sp * cy + sr * cp * sy
    z = sr * cp * cy - cr * sp * sy

    # return as (x, y, z, w)
    return np.array([x, y, z, w])

def sample_euler_upper_hemisphere(x_range=(np.pi/6*5, np.pi/6*7), y_range=(0, np.pi/2)):
    x = np.random.uniform(*x_range)
    y = np.random.uniform(-np.pi/4, np.pi/4)
    z = np.random.uniform(-np.pi/4, np.pi/4)
    return [x, y, z]

def sample_pose_on_sphere(distance=0.01, phi_range=(0, np.pi/2)):
    """
    Sample a random pose on a sphere.
    """
    # Randomly sample a point on the sphere around the object
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(phi_range[0], phi_range[1])
    x = distance * np.sin(phi) * np.cos(theta)
    y = distance * np.sin(phi) * np.sin(theta)
    z = distance * np.cos(phi)

    euler = sample_euler_upper_hemisphere()
    # print(f"Sampled euler angles: {euler}")
    quat = R.from_euler('xyz', euler, degrees=False).as_quat()

    # # Draw the xyz
    # point = [x, y, z]
    # self.addDebugPoint(point, color=[1, 0, 0], size=0.01)

    # Return the new pose with the same orientation as the object
    return np.array([x, y, z, quat[0], quat[1], quat[2], quat[3]])

class VersatileWaypointSampler:
    """
    A versatile waypoint sampler that can generate waypoints for various tasks.
    This class can be extended to support different types of tasks by implementing
    specific waypoint sampling methods.
    """
    def __init__(self):
        self.pick_and_place_sampler = PickAndPlaceWaypointSampler()
        self.move_to_target_sampler = MoveToTargetWaypointSampler()
        self.push_sampler = PushWaypointSampler()
        self.pull_sampler = PullWaypointSampler()
        self.pick_sampler = PickWaypointSampler()
    
    def sample_waypoints(self, task_type=None, object_ids=[0,1]):
        """
        Sample waypoints based on the specified task type.

        Args:
            task_type (str): The type of task for which to sample waypoints.
            object_ids (list): List of object IDs involved in the task.

        Returns:
            np.ndarray: An N×4 array of rows [object_id, x, y, z].
        """
        if task_type is None:
            # If no task type is specified, randomly select one
            task_type = np.random.choice(TASK_TYPES)
            print(f"Randomly selected task type: {task_type}")
        if task_type == 'pick_and_place':
            return self.pick_and_place_sampler.sample_waypoints(object_ids), 2
        elif task_type == 'move_to_target':
            return self.move_to_target_sampler.sample_waypoints(object_ids), 2
        elif task_type == 'push':
            return self.push_sampler.sample_waypoints(object_ids), 1
        elif task_type == 'pull':
            return self.pull_sampler.sample_waypoints(object_ids), 1
        elif task_type == 'pick':
            return self.pick_sampler.sample_waypoints(object_ids), 1
        else:
            raise ValueError(f"Unknown task type: {task_type}. Supported types are: {TASK_TYPES}")

class PickAndPlaceWaypointSampler:
    def __init__(self):
        pass

    def sample_waypoints(self, object_ids):
        rel_waypoints = []
        first_obj_id, second_obj_id = 0, 1
        # Pre-grasp pose
        pre_grasp_pose = sample_pose_on_sphere(distance=0.2, phi_range=(0, 0.3))
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), np.array(pre_grasp_pose)]))
        # Approach first object
        grasp_pose = sample_pose_on_sphere(distance=0.1)
        grasp_pose[3:] = pre_grasp_pose[3:]  # Keep the orientation from pre-grasp
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), np.array(grasp_pose)]))
        # Grasp first object
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), GRASP_CODE]))
        # Pre-place pose
        pre_place_pose = sample_pose_on_sphere(distance=0.2, phi_range=(0, 0.3))
        rel_waypoints.append(np.concatenate([np.array(second_obj_id).reshape(-1,), np.array(pre_place_pose)]))
        # Approach second object
        place_pose = sample_pose_on_sphere(distance=0.1)
        place_pose[3:] = pre_place_pose[3:]
        rel_waypoints.append(np.concatenate([np.array(second_obj_id).reshape(-1,), np.array(place_pose)]))
        # Release first object
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), RELEASE_CODE]))
        rel_waypoints = np.stack(rel_waypoints)
        return rel_waypoints

class MoveToTargetWaypointSampler:
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
        first_object_id, second_object_id = 0, 1
        
        # Assemble and return waypoints
        rel_waypoints = []
        sampled_pose = sample_pose_on_sphere(distance=0.1)
        rel_waypoints.append(np.concatenate([np.array([first_object_id]).reshape(-1,), sampled_pose]))
        rel_waypoints.append(np.concatenate([np.array([first_object_id]).reshape(-1,), GRASP_CODE]))
        sampled_pose = sample_pose_on_sphere(distance=0.2)
        rel_waypoints.append(np.concatenate([np.array([second_object_id]).reshape(-1,), sampled_pose]))
        rel_waypoints.append(np.concatenate([np.array([second_object_id]).reshape(-1,), RELEASE_CODE]))
        return np.stack(rel_waypoints)

class PushWaypointSampler:
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
        sampled_contact_point = sample_pose_on_sphere(distance=0.1, phi_range=(np.pi/2-0.3, np.pi/2))

        rel_direction = sampled_contact_point[:3].copy()
        rel_direction /= np.linalg.norm(rel_direction)  # Normalize the direction vector

        # Pre-push: stand-off before contact
        pre_push = 0.2 * rel_direction.copy()
        pre_push = np.concatenate([pre_push, sampled_contact_point[3:]])  # Keep orientation as (1, 0, 0, 0)

        # Post-push: endpoint after sliding the object along the push direction
        post_push = -0.3 * rel_direction.copy()
        post_push = np.concatenate([post_push, sampled_contact_point[3:]])  # Keep orientation as (1, 0, 0, 0)])

        # Assemble and return waypoints
        rel_waypoints = []
        rel_waypoints.append(np.concatenate([np.array([-1]).reshape(-1,), GRASP_CODE]))
        rel_waypoints.append(np.concatenate([np.array([object_id]), pre_push]))
        rel_waypoints.append(np.concatenate([np.array([object_id]), sampled_contact_point]))
        rel_waypoints.append(np.concatenate([np.array([object_id]).reshape(-1,), GRASP_CODE]))
        rel_waypoints.append(np.concatenate([np.array([object_id]), post_push]))
        return np.stack(rel_waypoints)

class PullWaypointSampler:
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
        sampled_contact_point = sample_pose_on_sphere(distance=0.1, phi_range=(np.pi/2-0.3, np.pi/2))

        rel_direction = sampled_contact_point[:3].copy()
        rel_direction /= np.linalg.norm(rel_direction)  # Normalize the direction vector

        # Pre-push: stand-off before contact
        post_pull = 0.3 * rel_direction.copy()
        post_pull = np.concatenate([post_pull, sampled_contact_point[3:]])  # Keep orientation as (1, 0, 0, 0)

        # Assemble and return waypoints
        rel_waypoints = []
        rel_waypoints.append(np.concatenate([np.array([object_id]), sampled_contact_point]))
        rel_waypoints.append(np.concatenate([np.array([object_id]).reshape(-1,), GRASP_CODE]))
        rel_waypoints.append(np.concatenate([np.array([object_id]), post_pull]))
        return np.stack(rel_waypoints)

class PickWaypointSampler:
    def __init__(self):
        pass

    def sample_waypoints(self, object_ids):
        rel_waypoints = []
        first_obj_id = 0
        # Pre-grasp pose
        pre_grasp_pose = sample_pose_on_sphere(distance=0.2, phi_range=(0, 0.3))
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), np.array(pre_grasp_pose)]))
        # Approach first object
        grasp_pose = sample_pose_on_sphere(distance=0.1)
        grasp_pose[3:] = pre_grasp_pose[3:]  # Keep the orientation from pre-grasp
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), np.array(grasp_pose)]))
        # Grasp first object
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), GRASP_CODE]))
        # Post-grasp pose
        post_grasp_pose = sample_pose_on_sphere(distance=0.2, phi_range=(0, 0.3))
        post_grasp_pose[3:] = grasp_pose[3:]
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), np.array(post_grasp_pose)]))
        rel_waypoints = np.stack(rel_waypoints)
        return rel_waypoints