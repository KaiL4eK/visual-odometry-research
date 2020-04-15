import numpy as np

# Based on https://github.com/Huangying-Zhan/kitti-odom-eval/blob/master/kitti_odometry.py

def rotation_error(pose_error):
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5*(a+b+c-1.0)
    rot_error = np.arccos(np.clip(d, -1.0, 1.0))
    return rot_error


def translation_error(pose_error):
    trans_error = np.linalg.norm(pose_error[:3,3])
    return trans_error


def compute_ATE(gt, pred):
    errs = gt[:, :3, 3] - pred[:, :3, 3]
    magns = np.linalg.norm(errs, axis=1)
    # TODO - sqrt and **2 required??
    ate = np.sqrt(np.mean(magns ** 2)) 
    return ate


def compute_RPE(gt, pred):
    trans_errors = []
    rot_errors = []
    for i in range(len(gt))[:-1]:
        gt1 = gt[i]
        gt2 = gt[i+1]
        gt_rel = np.linalg.inv(gt1) @ gt2

        pred1 = pred[i]
        pred2 = pred[i+1]
        pred_rel = np.linalg.inv(pred1) @ pred2
        
        rel_err = np.linalg.inv(gt_rel) @ pred_rel

        trans_errors.append(translation_error(rel_err))
        rot_errors.append(rotation_error(rel_err))
    rpe_trans = np.mean(trans_errors)
    rpe_rot = np.mean(rot_errors)
    return rpe_trans, rpe_rot
