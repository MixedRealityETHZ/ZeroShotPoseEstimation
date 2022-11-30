import numpy as np
from matplotlib import pyplot as plt
import itertools 
from utils_betti import *
from plotting import plot_3D_scene

if __name__ == '__main__':
    # The name of the dataset defines the names of input and output directories.
    dataset = 'Tiger2'
     # Select whether to save output images to files.
    save_output_images = True
     # Randomly use less images (messo se ci sono video troppo lunghi)
    random_downsample = False
    # Plot in 3D
    plot = True

    bbs, K, Ms_t, visibility = get_data(dataset, random_downsample)
    # Compute the number of frames and the number of objects for the current dataset from the size of the visibility matrix.
    n_frames = visibility.shape[0]
    n_objects = visibility.shape[1]

    ######################################
    # 2. Run the algorithm: estimate the #
    #    object ellipsoids for all the   #
    #    objects in the scene.           #
    ######################################

    estQs = compute_estimates(bbs, K, Ms_t, visibility)

    ##################################
    # 3. Get the points of the bbox  #
    # of the object with object_idx  #
    ##################################
    object_idx = 0
    while(object_idx >= estQs.shape[0] or object_idx < 0):
        print("Insert a valid object idx, possible values are: " + str(np.arange(0, estQs.shape[0])))
        object_idx = int(input("Enter your value: "))
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


    # Plot ellipsoids, parralepiped and camera poses in 3D.
    if plot:
        fig = plot_3D_scene(estQs, estQs, Ms_t, dataset, save_output_images, points)
        plt.show()
