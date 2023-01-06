"""
The following script are used to compute the distance between the predicted poses
and the ground truth poses
"""

import math
import numpy as np
import glob
import os.path as osp
import natsort

def matrix_to_euler(R):
  # Extract the yaw angle
  yaw = np.arctan2(R[1, 0], R[0, 0])
  
  # Extract the pitch angle
  pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
  
  # Extract the roll angle
  roll = np.arctan2(R[2, 1], R[2, 2])
  
  return roll, pitch, yaw

def get_pose_format(pose):
    R = pose[0:3,0:3]
    x, y, z = pose[0:3,-1]
    roll, pitch, yaw = matrix_to_euler(R)

    return  x, y, z, roll, pitch, yaw

def pose_distance(pose1, pose2):
  # Unpack the poses
  x1, y1, z1, roll1, pitch1, yaw1 = get_pose_format(pose1)
  x2, y2, z2, roll2, pitch2, yaw2 = get_pose_format(pose2)
  
  # Compute the position distance
  pos_dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
  
  # Compute the orientation distance
  orient_dist = math.sqrt((roll1 - roll2)**2 + (pitch1 - pitch2)**2 + (yaw1 - yaw2)**2)
  
  # Return the total distance
  return pos_dist, orient_dist

def load_gt_poses(data_dir):

    poses_dir = osp.join(data_dir, "poses")
    pose_list = glob.glob(poses_dir + "/*.txt", recursive=True)
    pose_list = natsort.natsorted(pose_list)
    return pose_list
