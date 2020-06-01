import numpy as np
from matplotlib import pyplot as plt

def plot_trajectories(poses_gt, poses_result):
    plot_keys = ["Ground Truth", "Ours"]
    fontsize_ = 12

    poses_dict = {}
    poses_dict["Ground Truth"] = poses_gt
    poses_dict["Ours"] = poses_result

    fig = plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')

    for key in plot_keys:
        pos_xz = []
        frame_idx_list = range(len(poses_dict["Ours"]))
        for frame_idx in frame_idx_list:
            # pose = np.linalg.inv(poses_dict[key][frame_idx_list[0]]) @ poses_dict[key][frame_idx]
            pose = poses_dict[key][frame_idx]
            pos_xz.append([pose[0, 3],  pose[2, 3]])
        pos_xz = np.asarray(pos_xz)
        plt.plot(pos_xz[:, 0],  pos_xz[:, 1], label=key)

    plt.legend(loc="upper right", prop={'size': fontsize_})
    plt.xticks(fontsize=fontsize_)
    plt.yticks(fontsize=fontsize_)
    plt.xlabel('x (m)', fontsize=fontsize_)
    plt.ylabel('z (m)', fontsize=fontsize_)
    plt.grid()
    plt.show()
    
    
def plot_trajectory(poses):
    fontsize_ = 12

#     fig = plt.figure()
#     ax = plt.gca()
#     ax.set_aspect('equal')

    plt.plot(poses[:,0,3],  poses[:,2,3])
    plt.xticks(fontsize=fontsize_)
    plt.yticks(fontsize=fontsize_)
    plt.xlabel('x (m)', fontsize=fontsize_)
    plt.ylabel('z (m)', fontsize=fontsize_)
    plt.grid()
    plt.show()
    