import numpy as np
import os

DATA_DIR = './eval_results/test_euroc/exp_bs=8_lr=3e-6_lw=(4,0.1,2,0.1)'
method = 'flowNet'

timestamps_dir = os.path.join(DATA_DIR, 'timestamp.txt')
groundtruth_dir = os.path.join(DATA_DIR, 'gt_pose.txt')

timestamps = np.loadtxt(timestamps_dir)
groundtruth = np.loadtxt(groundtruth_dir)

length_gt = len(groundtruth)
groundtruth_pose = np.concatenate((timestamps[:length_gt].reshape(-1, 1), groundtruth), axis=1)
np.savetxt(os.path.join(DATA_DIR, 'groundtruth.txt'), groundtruth_pose)

method_dir = os.path.join(DATA_DIR, '1_'+method)
estimated_poses_dir = [os.path.join(method_dir, f) for f in os.listdir(method_dir) if f.endswith('.txt') and 'pose' in f]

for i, pose_dir in enumerate(estimated_poses_dir):
    estimated_pose = np.loadtxt(pose_dir)
    length_ep = len(estimated_pose)
    estimated_pose = np.concatenate((timestamps[:length_ep].reshape(-1, 1), estimated_pose), axis=1)
    save_filename = 'estimated_' + pose_dir.split('/')[-1].split('_')[0] + '.txt'
    np.savetxt(os.path.join(method_dir, save_filename), estimated_pose)

