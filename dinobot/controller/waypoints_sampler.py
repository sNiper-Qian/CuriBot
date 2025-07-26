import numpy as np

GRASP_CODE = np.ones(7)
RELEASE_CODE = np.zeros(7)
TASK_TYPES = ['pick_and_place', 'move_to_target', 'push', 'pull', 'pick']

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

    # Orientation is currently always (1, 0, 0, 0)
    # TODO
    quat = [1, 0, 0, 0]

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
        sampled_pose = sample_pose_on_sphere(distance=0.2, phi_range=(0, 0.3))
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), np.array(sampled_pose)]))
        # Approach first object
        sampled_pose = sample_pose_on_sphere(distance=0.1)
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), np.array(sampled_pose)]))
        # Grasp first object
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), GRASP_CODE]))
        # Pre-place pose
        sampled_pose = sample_pose_on_sphere(distance=0.2, phi_range=(0, 0.3))
        rel_waypoints.append(np.concatenate([np.array(second_obj_id).reshape(-1,), np.array(sampled_pose)]))
        # Approach second object
        sampled_pose = sample_pose_on_sphere(distance=0.1)
        rel_waypoints.append(np.concatenate([np.array(second_obj_id).reshape(-1,), np.array(sampled_pose)]))
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
        pre_push = np.concatenate([pre_push, np.array([1, 0, 0, 0])])  # Keep orientation as (1, 0, 0, 0)

        # Post-push: endpoint after sliding the object along the push direction
        post_push = -0.3 * rel_direction.copy()
        post_push = np.concatenate([post_push, np.array([1, 0, 0, 0])])

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
        post_pull = np.concatenate([post_pull, np.array([1, 0, 0, 0])])  # Keep orientation as (1, 0, 0, 0)

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
        sampled_pose = sample_pose_on_sphere(distance=0.2, phi_range=(0, 0.3))
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), np.array(sampled_pose)]))
        # Approach first object
        sampled_pose = sample_pose_on_sphere(distance=0.1)
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), np.array(sampled_pose)]))
        # Grasp first object
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), GRASP_CODE]))
        # Post-grasp pose
        sampled_pose = sample_pose_on_sphere(distance=0.2, phi_range=(0, 0.3))
        rel_waypoints.append(np.concatenate([np.array(first_obj_id).reshape(-1,), np.array(sampled_pose)]))
        rel_waypoints = np.stack(rel_waypoints)
        return rel_waypoints