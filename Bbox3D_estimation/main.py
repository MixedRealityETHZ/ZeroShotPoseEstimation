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
            pose = np.transpose(np.loadtxt(f_input)[:3, :])
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
            corpus.append([int(number) for number in line.split(",")])
    return np.array(corpus)


###########################################
# 1. Set the parameters for the algorithm #
#    and load the input data.             #
###########################################
# Select the dataset to be used.
# The name of the dataset defines the names of input and output directories.
dataset = "Aldoma"

# Select whether to save output images to files.
save_output_images = True

# Randomly use less images (messo se ci sono video troppo lunghi)
random_downsample = False

# Plot in 3D
plot = False
if dataset != "Aldoma":
    PATH = f"data/{dataset}"
    box_list = glob.glob(os.path.join(os.getcwd(), f"{PATH}/bounding_boxes", "*.txt"))
    poses_list = glob.glob(os.path.join(os.getcwd(), f"{PATH}/poses_ba", "*.txt"))
    intrinsics = f"{PATH}/intrinsics.txt"
else:
    PATH = f"data/{dataset}"
    box_list = glob.glob(os.path.join(os.getcwd(), f"{PATH}/bounding_boxes.npy"))
    poses_list = glob.glob(os.path.join(os.getcwd(), f"{PATH}/camera_poses.npy.npy"))
    intrinsics = f"{PATH}/intrinsics.npy"

bbs = read_list_box(box_list)
Ms_t = read_list_poses(poses_list)
visibility = np.ones((bbs.shape[0], 1))

if random_downsample:
    randomline = np.random.choice(bbs.shape[0], 10)
    visibility[randomline, :] = 1

with open(intrinsics) as f:
    intr = f.readlines()
    K = np.array(
        [
            [float(intr[0]), 0, float(intr[2])],
            [0, float(intr[1]), float(intr[3])],
            [0, 0, 1],
        ]
    )


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
mins = [c - ax / 2 for (ax, c) in zip(axes, centre)]
maxs = [c + ax / 2 for (ax, c) in zip(axes, centre)]

# Coordinates of the points mins and maxs
points = np.array(list(itertools.product(*zip(mins, maxs))))

# Points in the camera frame
points = np.dot(points, R.T)


# Plot ellipsoids and camera poses in 3D.
plot = True
if plot:

    plot_3D_scene(
        estQs=estQs,
        gtQs=estQs,
        Ms_t=Ms_t,
        dataset=dataset,
        save_output_images=save_output_images,
    )
    plt.show()
