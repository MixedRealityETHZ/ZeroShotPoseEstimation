import numpy as np
from matplotlib import pyplot as plt
import itertools 
from .utils import compute_estimates, dual_quadric_to_ellipsoid_parameters
from .plotting import plot_3D_scene

class Detector3D():
    def __init__(self, K) -> None:
        self.K = K
        self.bboxes = None
        self.poses = None


    def add_view(self, bbox_t: np.ndarray, pose_t: np.ndarray):
        if self.bboxes is None:
            self.bboxes = bbox_t
        else:
            self.bboxes = np.stack(self.bboxes, bbox_t)
        
        if self.poses is None:
            self.poses = pose_t
        else:
            self.poses = np.stack(self.poses, pose_t)


    def detect_3D_box(self):
        object_idx = 0
        visibility = np.ones_like(self.poses)
        estQs = compute_estimates(self.bboxes, self.K, self.poses, visibility)
        # while(object_idx >= estQs.shape[0] or object_idx < 0):
        #     print("Insert a valid object idx, possible values are: " + str(np.arange(0, estQs.shape[0])))
        #     object_idx = int(input("Enter your value: "))
        
        centre, axes, R = dual_quadric_to_ellipsoid_parameters(estQs[object_idx])

        # Possible coordinates
        mins = [-ax for (ax) in axes]
        maxs = [ax for (ax) in axes]

        # Coordinates of the points mins and maxs
        points = np.array(list(itertools.product(*zip(mins, maxs))))

        # Points in the camera frame
        points = np.dot(points, R.T)

        # Shift correctly the parralelepiped
        points[:, 0:3] = np.add(centre[None, :], points[:, :3],)
        
        self.points = points

    def save_3D_box(self, data_root):
        np.savetxt(data_root + '/box3d_corners.txt', self.points, delimiter=' ')


    
