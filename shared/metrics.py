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
    if isinstance(gt,list):
        gt = np.array(gt)
    if isinstance(pred,list):
        pred = np.array(pred)

    errs = gt[:, :3, 3] - pred[:, :3, 3]
    magns = np.linalg.norm(errs, axis=1)
    # TODO - sqrt and **2 required??
    ate = np.sqrt(np.mean(magns**2))
    return ate


def compute_RPE(gt, pred):
    if isinstance(gt,list):
        gt = np.array(gt)
    if isinstance(pred,list):
        pred = np.array(pred)

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


def trajectory_distances(poses):
    diffs = np.diff(poses[:,:3,3], axis=0)
    deltas = np.linalg.norm(diffs, axis=1)
    dists = np.cumsum([0, *deltas])
    return dists


def last_frame_from_segment_length(dists, first_frame, length):
    for i in range(first_frame, len(dists)):
        if dists[i] > (dists[first_frame] + length):
            return i
    return -1


def calc_sequence_errors(poses_gt, poses_result):
    err = []
    dists = trajectory_distances(poses_gt)
    step_size = 10
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]

    for first_frame_idx in range(0, len(poses_gt), step_size):
        for len_ in lengths:
            last_frame_idx = last_frame_from_segment_length(
                                    dists, first_frame_idx, len_
                                    )
            if last_frame_idx == -1:
                continue

            pose_delta_gt = np.dot(
                                poses_gt[last_frame_idx],
                                np.linalg.inv(poses_gt[first_frame_idx])
                                )
            pose_delta_result = np.dot(
                                    poses_result[last_frame_idx],
                                    np.linalg.inv(poses_result[first_frame_idx])
                                    )
            pose_error = np.dot(
                            pose_delta_gt,
                            np.linalg.inv(pose_delta_result)
                            )

            r_err = rotation_error(pose_error)
            t_err = translation_error(pose_error)

            # compute speed
            num_frames = last_frame_idx - first_frame_idx + 1.0
            speed = len_/(0.1*num_frames)

            err.append(
                (first_frame_idx, r_err/len_, t_err/len_, len_, speed)
            )
    return np.array(err)


def compute_segment_error(seq_errs):
    avg_segment_errs = {}
    lengths = np.unique(seq_errs[:,3])

    for len_ in lengths:
        data = seq_errs[seq_errs[:,3] == len_]
        avg = np.mean(data[:, 1:3], axis=0)
        avg_segment_errs[len_] = avg

    return avg_segment_errs


def compute_overall_err(seq_errs):
    if seq_errs.shape[0] > 0:
        return np.mean(seq_errs[:, 1:3], axis=0)
    else:
        return 0, 0
