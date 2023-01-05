import os
import glob
import numpy as np
from src.bbox_3D_estimation.utils import read_list_poses_orig
import re
from scipy.spatial.transform import Rotation as R



r = R.from_quat([0.3535534, 0.3535534, 0.1464466, 0.8535534])
print(r.as_matrix())

regex = re.compile('[^0-9]')

DIR = "data/costum_datasets/test/demo_bottle/bottle-1/"

poses = sorted(glob.glob(f"{DIR}backup/poses/*.txt"))
names = []
for pose in poses:
    name = regex.sub("", pose)[1:]
    names.append(name)
poses = read_list_poses_orig(poses)

M = np.empty((4, 4))
# M[:3, :3] = np.array([[  1.0000000, 0.0000000,  0.0000000],
#                       [  0.0000000, 1.0000000,  0.0000000],
#                       [  0.0000000, 0.0000000, 1.0000000 ]])
M[:3, :3] = r.as_matrix()
M[:3, 3] =  np.array([1, 1, 0])
M[3, :] = [0, 0, 0, 1]


def ruf_to_flu(pose):
    rotquat_ruf = R.from_matrix(pose[:3,:3]).as_quat()
    rotquat_flu = np.array([-rotquat_ruf[2], rotquat_ruf[0], -rotquat_ruf[1], rotquat_ruf[3]])
    rotmat_flu = R.from_quat(rotquat_flu).as_matrix()

    transl_ruf = pose[:3, 3]
    transl_flu = np.array([transl_ruf[2], -transl_ruf[0], transl_ruf[1]])

    new_pose = np.eye(4)
    new_pose[:3,:3] = rotmat_flu @ np.array([0,0,1,0,1,0,-1,0,0]).reshape((3,3))
    new_pose[:3, 3] = transl_flu 

    return new_pose

shifted_poses = []
for pose in poses:

    
    # pose = np.linalg.inv(pose)
    # pose = np.dot(M, pose)

    # Extract rotation matrix and translation vector from left-handed pose T
    # R = pose[:3, :3]
    # t = pose[:3, 3]

    # # Negate last element of translation vector
    # t[0] = t[2]
    # t[2] = t[0]

    # # Convert the pose to a right-handed coordinate system
    # R_right = np.array([[R[0,2], R[0,1], R[0,0]],
    #                     [R[1,2], R[1,1], R[1,0]],
    #                     [R[2,2], R[2,1], R[2,0]]])

    # # Construct right-handed pose T'
    # T_prime = np.eye(4)
    # T_prime[:3, :3] = R_right
    # T_prime[:3, 3] = t

    
    # inverted = T_prime #np.dot(T_prime, pose)
    # T_prime = pose
    # pose = ruf_to_flu(pose)

    original = np.linalg.inv(pose)

    shifted_poses.append(original)


shift_pose_dir = f"{DIR}poses/"
os.makedirs(shift_pose_dir, exist_ok=True)
for pose, name in zip(shifted_poses, names):
    np.savetxt(f"{shift_pose_dir}{name}.txt", pose, delimiter=" ")
