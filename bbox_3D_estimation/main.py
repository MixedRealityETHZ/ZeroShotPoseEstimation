"""main - Localisation from Detections - Main.

###################
0. Introduction. #
###################
Conventions: variable names such as Ms_t indicate that there are multiple M matrices (Ms)
which are transposed (_t) respect to the canonical orientation of such data.
Prefixes specify if one variable refers to input data, estimates etc.: inputCs, estCs, etc.

Variable names:
C - Ellipse in dual form [3x3].
Q - Quadric/Ellipsoid in dual form [4x4], in the World reference frame.
K - Camera intrinsics [3x3].
M - Pose matrix: transforms points from the World reference frame to the Camera reference frame [3x4].
P - Projection matrix = K * M [3*4].

A note on the visibility information: when false, it might mean that either the object is not visible in the image,
or that the detector failed. For these cases the algorithm does not visualise the estimated, nor the GT ellipses.

If one object is not detected in at least 3 frames, it is ignored. The values of the corresponding ellipsoid and
ellipses are set to NaN, so the object is never visualised.
"""

import os
import glob
import pickle
import itertools
import numpy as np
from matplotlib import pyplot as plt
from plotting import plot_est_and_gt_ellipses_on_images, plot_3D_scene
from lfd import compute_estimates, dual_quadric_to_ellipsoid_parameters


# Utilities
def read_list_poses(list):
    for idx, file_path in enumerate(list):
        with open(file_path) as f_input:
            orig_pose = np.loadtxt(f_input)
            orig_pose = np.linalg.inv(orig_pose)
            pose = np.transpose(orig_pose[:3, :])
            #pose = np.transpose(np.loadtxt(f_input)[:3, :])
            if idx == 0:
                poses = pose
            else:
                poses = np.concatenate((poses, pose), axis=0)
    return poses


def read_list_box(list):
    corpus = []
    for file_path in list:
        with open(file_path) as f_input:
            line = f_input.read()
            corpus.append([float(number) for number in line.split(",")])
    return np.array(corpus)


###########################################
# 1. Set the parameters for the algorithm #
#    and load the input data.             #
###########################################
# Select the dataset to be used.
# The name of the dataset defines the names of input and output directories.
dataset = "tiger"

# Select whether to save output images to files.
save_output_images = True

# Randomly use less images (messo se ci sono video troppo lunghi)
random_downsample = False

if dataset != "Aldoma":
    PATH = f"data/{dataset}"
    box_list = sorted(glob.glob(os.path.join(os.getcwd(), f"{PATH}/bounding_boxes", "*.txt")))
    poses_list = sorted(glob.glob(os.path.join(os.getcwd(), f"{PATH}/poses_ba", "*.txt")))
    intrinsics = f"{PATH}/intrinsics.txt"
    bbs = read_list_box(box_list)
    Ms_t = read_list_poses(poses_list)
    GT_bb = np.loadtxt(f"{PATH}/box3d_corners.txt")
    visibility = np.ones((bbs.shape[0], 1))
    with open(intrinsics) as f:
        intr = f.readlines()
        K = np.array(
            [
                [float(intr[0]), 0, float(intr[2])],
                [0, float(intr[1]), float(intr[3])],
                [0, 0, 1],
            ]
        )
        # K = np.array(
        #     [
        #         [float(1540), 0, float(719)],
        #         [0, float(1540), float(963)],
        #         [0, 0, 1],
        #     ]
        # )
        
else:
    bbs = np.load('data/{:s}/bounding_boxes.npy'.format(dataset))  
    K = np.load('data/{:s}/intrinsics.npy'.format(dataset))
    Ms_t = np.load('data/{:s}/camera_poses.npy'.format(dataset)) 
    visibility = np.load('data/{:s}/visibility.npy'.format(dataset)) 


if random_downsample:
    randomline = np.random.choice(bbs.shape[0], 100)
    visibility[randomline, :] = 1


# Compute the number of frames and the number of objects
# for the current dataset from the size of the visibility matrix.

n_frames = visibility.shape[0]
n_objects = visibility.shape[1]

######################################
# 2. Run the algorithm: estimate the #
#    object ellipsoids.              #
######################################

inputCs, estCs, estQs = compute_estimates(bbs, K, Ms_t, visibility)

#############################
# 3. I punti del 3D BB. #
#############################
centre, axes, R = dual_quadric_to_ellipsoid_parameters(estQs[0])

# Possible coordinates
mins = [-ax for (ax) in axes]
maxs = [ax for (ax) in axes]

# Coordinates of the points mins and maxs
points = np.array(list(itertools.product(*zip(mins, maxs))))

# Points in the camera frame
points = np.dot(points, R.T)

# Shift correctly the parralelepiped
points[:, 0:3] = np.add(centre[None, :], points[:, :3],)

#print(points)

# Plot ellipsoids and camera poses in 3D.
plot = True
if plot:

    plot_3D_scene(
        estQs=estQs,
        gtQs=estQs,
        Ms_t=Ms_t,
        dataset=dataset,
        save_output_images=save_output_images,
        points=points,
        GT_points=GT_bb        
    )
    plt.show()
