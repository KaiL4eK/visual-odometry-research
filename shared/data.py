import os
import cv2

import numpy as np
import quaternion as quat


class KITTIData(object):
    def __init__(self, dataset_dir, sequence_id='00'):
        SEQUENCE_DIR = os.path.join(dataset_dir, 'sequences', sequence_id)
        POSES_DIR = os.path.join(dataset_dir, 'poses')

        POSE_PATH = os.path.join(POSES_DIR, f'{sequence_id}.txt')
        TIMES_PATH = os.path.join(SEQUENCE_DIR, 'times.txt')
        CALIB_PATH = os.path.join(SEQUENCE_DIR, 'calib.txt')
        
        self.poses = self._load_poses(POSE_PATH)
        self.times = self._load_times(TIMES_PATH)
        
        self._load_calib(CALIB_PATH)
        
#         print('Projections:')
#         for name, intr in self.projections.items():
#             print(f'{name}:\n{intr}')
#         print('Intrinsics:')
#         for name, intr in self.intricsics.items():
#             print(f'{name}:\n{intr}')
#         print('Extrinsics to cam0:')
#         for name, extr in self.cam0_extrinsics.items():
#             print(f'{name}:\n{extr}')
#         print('Extrinsics to velo:')
#         for name, extr in self.velo_extrinsics.items():
#             print(f'{name}:\n{extr}')
#         print('Baselines:')
#         print(self.baselines)

        
        self.LEFT_GRAY_IMAGES_DIR = os.path.join(SEQUENCE_DIR, 'image_0')
        self.RIGHT_GRAY_IMAGES_DIR = os.path.join(SEQUENCE_DIR, 'image_1')
    
        self.LEFT_IMAGES_DIR = os.path.join(SEQUENCE_DIR, 'image_2')
        self.RIGHT_IMAGES_DIR = os.path.join(SEQUENCE_DIR, 'image_3')

        self.images = [fname for fname in os.listdir(self.LEFT_IMAGES_DIR) if fname.endswith('.png')]
        
        print(f'Sequence {sequence_id} length: {len(self.poses)}')
        
        # Sanity check!
        for i in range(len(self.poses)):
            fname = self._get_image_fname(i)
            if fname not in self.images:
                raise Exception(f'File with name {fname} not exists in {IMAGES_DIR}')
        # After this check we can use idx to generate fpaths
        
    def _get_image_fname(self, idx):
        return f'{idx:06}.png'
        
    def _load_times(self, fpath):
        times_data = np.fromfile(fpath, sep='\n')
        return times_data
    
    def _load_poses(self, fpath):
        poses_data = np.fromfile(fpath, sep=' ')
        poses_data = poses_data.reshape((-1, 3, 4))
        # Convert to 4x4 matrices
        last_row = np.array([[[0,0,0,1]]])
        last_rows = np.repeat(last_row, axis=0, repeats=poses_data.shape[0])
        poses_data = np.hstack((poses_data, last_rows))
        return poses_data
    
    # Based on
    # https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/odometry.py#L145
    def _load_calib(self, fpath):
        filedata = {}
        with open(fpath) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                
                name, info = line.split(':')
                calib_data = np.fromstring(info, sep=' ')
                filedata[name] = calib_data.reshape(3, 4)

        self.projections = {}
        self.projections['P0'] = filedata['P0']
        self.projections['P1'] = filedata['P1']
        self.projections['P2'] = filedata['P2']
        self.projections['P3'] = filedata['P3']
         
        self.intricsics = {}
        self.intricsics['K_cam0'] = filedata['P0'][0:3, 0:3]
        self.intricsics['K_cam1'] = filedata['P1'][0:3, 0:3]
        self.intricsics['K_cam2'] = filedata['P2'][0:3, 0:3]
        self.intricsics['K_cam3'] = filedata['P3'][0:3, 0:3]
        
        self.cam0_extrinsics = {}
        self.cam0_extrinsics['T1'] = np.eye(4)
        self.cam0_extrinsics['T1'][0, 3] = filedata['P1'][0,3] / filedata['P1'][0,0]
        self.cam0_extrinsics['T2'] = np.eye(4)
        self.cam0_extrinsics['T2'][0, 3] = filedata['P2'][0,3] / filedata['P2'][0,0]
        self.cam0_extrinsics['T3'] = np.eye(4)
        self.cam0_extrinsics['T3'][0, 3] = filedata['P3'][0,3] / filedata['P3'][0,0]
        
        self.velo_extrinsics = {}
        self.velo_extrinsics['T0'] = np.reshape(filedata['Tr'], (3, 4))
        self.velo_extrinsics['T0'] = np.vstack([self.velo_extrinsics['T0'], [0, 0, 0, 1]])
        self.velo_extrinsics['T1'] = self.cam0_extrinsics['T1'].dot(self.velo_extrinsics['T0'])
        self.velo_extrinsics['T2'] = self.cam0_extrinsics['T2'].dot(self.velo_extrinsics['T0'])
        self.velo_extrinsics['T3'] = self.cam0_extrinsics['T3'].dot(self.velo_extrinsics['T0'])        
            
        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(self.velo_extrinsics['T0']).dot(p_cam)
        p_velo1 = np.linalg.inv(self.velo_extrinsics['T1']).dot(p_cam)
        p_velo2 = np.linalg.inv(self.velo_extrinsics['T2']).dot(p_cam)
        p_velo3 = np.linalg.inv(self.velo_extrinsics['T3']).dot(p_cam)

        self.baselines = {}
        self.baselines['rgb'] = np.linalg.norm(p_velo3 - p_velo2)
        self.baselines['gray'] = np.linalg.norm(p_velo1 - p_velo0)
        
    def get_poses(self):
        return self.poses
        
    def __len__(self):
        return len(self.poses)-1

    def _get_transform(self, idx):
        c_idx = idx
        n_idx = idx+1
        
        c_pose = self.poses[c_idx]
        n_pose = self.poses[n_idx]

        local_dtrans = np.linalg.inv(c_pose) @ n_pose[:, 3]

        quat_c = quat.from_rotation_matrix(c_pose[:3,:3])
        quat_n = quat.from_rotation_matrix(n_pose[:3,:3])
        quat_t = quat_c.inverse() * quat_n

        gt_quat_t_ar = quat.as_float_array(quat_t).astype(np.float32)
        gt_trans = local_dtrans[:3].astype(np.float32)
        return gt_quat_t_ar, gt_trans
    
    def _get_transform_mtrx(self, idx):
        c_idx = idx
        n_idx = idx+1
        
        c_pose = self.poses[c_idx]
        n_pose = self.poses[n_idx]

        local_dtrans = np.linalg.inv(c_pose) @ n_pose[:, 3]

        quat_c = quat.from_rotation_matrix(c_pose[:3,:3])
        quat_n = quat.from_rotation_matrix(n_pose[:3,:3])
        quat_t = quat_c.inverse() * quat_n
        
        transform = np.eye(4)
        
        transform[:3, :3] = quat.as_rotation_matrix(quat_t)
        transform[:3, 3] = local_dtrans[:3]

        return transform
    
    def get_poses_transform(self, idx):
        c_idx = idx
        n_idx = idx+1
        
        c_pose = self.poses[c_idx]
        n_pose = self.poses[n_idx]

        return n_pose @ np.linalg.inv(c_pose)

    def _intrinsics_dict(self, mtrx):
        return {
            'cx': mtrx[0,2],
            'cy': mtrx[1,2],
            'fx': mtrx[0,0],
            'fy': mtrx[1,1]
        }
    
    def get_С_matrix(self, color=False):
        if color:
            return self.intricsics['K_cam2'], self.intricsics['K_cam3']
        else:
            return self.intricsics['K_cam0'], self.intricsics['K_cam1']
    
    def get_intrinsics_dicts(self, color=False):
        if color:
            left = self._intrinsics_dict(self.intricsics['K_cam2'])
            right = self._intrinsics_dict(self.intricsics['K_cam3'])
        else:
            left = self._intrinsics_dict(self.intricsics['K_cam0'])
            right = self._intrinsics_dict(self.intricsics['K_cam1'])
        return left, right
        
    def get_left_Q_matrix(self, color=False):
        if color:
            intr = self._intrinsics_dict(self.intricsics['K_cam2'])
            # NOTE - Negative baseline, projecting right to left
            baseline = -self.baselines['rgb']
        else:
            intr = self._intrinsics_dict(self.intricsics['K_cam0'])
            # NOTE - Negative baseline, projecting right to left
            baseline = -self.baselines['gray']
        
        Q = np.array([
            [1, 0, 0, -intr['cx']],
            [0, 1, 0, -intr['cy']],
            [0, 0, 0, intr['fx']],
            [0, 0, -1/baseline, 0]
        ])
        
        return Q

    def get_images(self, idx, color=False):
        fname = self._get_image_fname(idx)
        if color:
            fname = self._get_image_fname(idx)
            left_img_fpath = os.path.join(self.LEFT_IMAGES_DIR, fname)
            right_img_fpath = os.path.join(self.RIGHT_IMAGES_DIR, fname)
            l_img = cv2.imread(left_img_fpath)
            l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB)
            r_img = cv2.imread(right_img_fpath)
            r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        else:
            left_img_fpath = os.path.join(self.LEFT_GRAY_IMAGES_DIR, fname)
            right_img_fpath = os.path.join(self.RIGHT_GRAY_IMAGES_DIR, fname)
            l_img = cv2.imread(left_img_fpath, cv2.IMREAD_GRAYSCALE)
            r_img = cv2.imread(right_img_fpath, cv2.IMREAD_GRAYSCALE)
        return l_img, r_img
            
    def __getitem__(self, idx):
        c_idx = idx
        n_idx = idx+1
        
        c_pose = self.poses[c_idx]
        n_pose = self.poses[n_idx]

        c_img_fpath = os.path.join(
            self.IMAGES_DIR, 
            self._get_image_fname(c_idx)
        )
        n_img_fpath = os.path.join(
            self.IMAGES_DIR, 
            self._get_image_fname(n_idx)
        )
        
        c_img = cv2.imread(c_img_fpath)
        c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB)
        n_img = cv2.imread(n_img_fpath)
        n_img = cv2.cvtColor(n_img, cv2.COLOR_BGR2RGB)
        gt_quat_t_ar, gt_trans = self._get_transform(idx)
        return c_img, n_img, gt_quat_t_ar, gt_trans

import networkx as nx
from networkx.algorithms.approximation.clique import max_clique
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
    
    
class VisualOdometry():
    def __init__(self):
        block = 9
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16*5, 
            blockSize=block,
#             P1=block*block*8,
#             P2=block*block*32,
#             disp12MaxDiff=1,
            preFilterCap=1,
            uniquenessRatio=5,
            speckleWindowSize=200,
            speckleRange=8
        )
        
    def get_next_pose(self, local_transform, c_pose):
        n_pose = np.eye(4)
        n_pose[:3,:3] = local_transform[:3,:3] @ c_pose[:3,:3]
        n_pose[:3,3] = c_pose[:3,3] + c_pose[:3,:3] @ local_transform[:3,3] 
        return n_pose

    def _get_next_pose(self, local_transform, c_pose):
        n_pose = np.eye(4)
        quat_t = quat.from_rotation_matrix(local_transform[:3,:3])
        quat_c = quat.from_rotation_matrix(c_pose[:3,:3])
        n_pose[:3, :3] = quat.as_rotation_matrix(quat_c * quat_t)
        n_pose[:, 3] = c_pose @ local_transform[:,3]
        return n_pose
    
    def reproject_3d_to_2d(self, pnts3d, P):
        # shape = [n_points x 3]
        # pnts3d_hmgns - Homogenous representation
        pnts3d_hmgns = np.ones((pnts3d.shape[0], pnts3d.shape[1]+1))
        pnts3d_hmgns[:,:3] = pnts3d
        
        pnts2d = P @ pnts3d_hmgns.T
        pnts2d = pnts2d.T
        pnts2d /= pnts2d[:,2:3] # To keep array
        pnts2d = pnts2d[:,:2].astype(int)
        
        for p in np.hstack((pnts3d, pnts2d)):
            if np.abs(p[-1]) > 10000:
                print(p)
        
        return pnts2d
    
    def reproject_2d_to_3d_points(self, feats, depth_frame, min_z=1, max_z=100):
        points = []
        feats = np.around(feats).astype(int)
        for ft in feats:
            pnt = depth_frame[ft[1], ft[0]]
            points.append(pnt)

        points = np.array(points)    
        ft_idxs = (points[:,2] > min_z) & (points[:,2] < max_z) & np.any(np.isfinite(points), axis=1)

        # All points, valid idxs
        return points, ft_idxs
    
    def transform_3d(self, T, pnts3d):
        pnts3d_hmgns = np.ones((pnts3d.shape[0], pnts3d.shape[1]+1))
        pnts3d_hmgns[:,:3] = pnts3d
        pred_pnts_3d = T @ pnts3d_hmgns.T
        pred_pnts_3d = pred_pnts_3d.T[:,:3]
        return pred_pnts_3d
    
    def process_depth(self, l_img, r_img, Q):
        if len(l_img.shape) == 3 and l_img.shape[2] > 1:
            l_img = cv2.cvtColor(l_img, cv2.COLOR_RGB2GRAY)
            r_img = cv2.cvtColor(r_img, cv2.COLOR_RGB2GRAY)

        l_disp = self.left_matcher.compute(l_img, r_img)        
        disp = l_disp.astype(np.float32) / 16.0
    
        depth_frame = cv2.reprojectImageTo3D(disp, Q)
        return depth_frame
    
    def max_clique_filter(self, c_pnts, n_pnts, dist_thrs=0.1, min_points=6):
        num_points = c_pnts.shape[0]
        graph = nx.Graph()
        graph.add_nodes_from(list(range(num_points)))

        if c_pnts.shape[0] < min_points:
            raise Exception('Too low count of points')

        clique_len = 0
        while clique_len < min_points:
            for i in range(num_points):
                diff_1 = c_pnts[i,:] - c_pnts
                diff_2 = n_pnts[i,:] - n_pnts
                dist_1 = np.linalg.norm(diff_1, axis=1)
                dist_2 = np.linalg.norm(diff_2, axis=1)
                diff = np.abs(dist_2 - dist_1)
                wIdx = np.where(diff < dist_thrs)
                for i_w in wIdx[0]:  
                    graph.add_edge(i, i_w)

            cliques = nx.algorithms.find_cliques(graph)
            _clique = max_clique(graph)
            clique_len = len(_clique)
            dist_thrs *= 2
        
        idxs = list(_clique)    
        return idxs, dist_thrs
    
    def _get_features_FAST(self, img):
        fe = cv2.FastFeatureDetector_create()
        feats = fe.detect(img)
        if len(feats) == 0:
            return None
        
        feats = [f.pt for f in feats]
        feats = np.array(feats)
        
        x_idxs = (feats[:,0] > 0) & (feats[:,0] < img.shape[1])
        y_idxs = (feats[:,1] > 0) & (feats[:,1] < img.shape[0])
        idxs = x_idxs & y_idxs
        
        feats = feats[idxs].astype(np.float32)
        return feats
    
    def _get_features_harris(self, img):
        feature_params = dict(maxCorners=20,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=3)
        
        feats = cv2.goodFeaturesToTrack(img, mask=None, **feature_params)
        if feats is None:
            return None
        
        x_idxs = (feats[:,:,0] > 0) & (feats[:,:,0] < img.shape[1])
        y_idxs = (feats[:,:,1] > 0) & (feats[:,:,1] < img.shape[0])
        idxs = x_idxs & y_idxs
        
        feats = feats[idxs].astype(np.float32)
        return feats
        
    def get_features(self, c_img, n_img):
        img_sz = c_img.shape[:2]
        tile_sz = np.array([100, 100])
        
        rate_sz = img_sz // tile_sz
        
        offset = (img_sz - rate_sz * tile_sz)/2
        offset = offset.astype(int)
       
        if len(c_img.shape) == 3 and c_img.shape[2] > 1:
            c_img = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)
            n_img = cv2.cvtColor(n_img, cv2.COLOR_RGB2GRAY)
        
        c_feats = []
        for y in range(offset[0], img_sz[0], tile_sz[0]):
            for x in range(offset[1], img_sz[1], tile_sz[1]):
                feats = self._get_features_harris(c_img[y:y+tile_sz[0], x:x+tile_sz[1]])
#                 feats = self._get_features_FAST(c_img[y:y+tile_sz[0], x:x+tile_sz[1]])
                if feats is None:
                    continue
                
                feats += np.array([x, y])
                c_feats.extend([f for f in feats])
        
        c_feats = np.array(c_feats)
        
        lk_params = dict(winSize=(15, 15),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

        n_feats, st, err = cv2.calcOpticalFlowPyrLK(c_img, n_img, c_feats, None, **lk_params)
        n_feats[:,0] = np.clip(n_feats[:,0], 0, img_sz[1]-1)
        n_feats[:,1] = np.clip(n_feats[:,1], 0, img_sz[0]-1)
        
        idxs = (st==1).flatten()
        c_feats = c_feats[idxs]
        n_feats = n_feats[idxs]

        return c_feats.astype(np.float32), n_feats.astype(np.float32)
    
    def rot_transform_matrix(x):
        transform = np.eye(4)
        transform[:3,:3] = R.from_rotvec(x[:3]).as_matrix()
        transform[:3, 3] = x[3:]
        return transform
    
    def _get_transform_PnP(self, c_pnts3d, n_pnts3d, C_mat):
        c_pnts2d, _ = cv2.projectPoints(c_pnts3d, np.zeros(3), np.zeros(3), C_mat, distCoeffs=None)
        retval, rvec, tvec = cv2.solvePnP(n_pnts3d, c_pnts2d, cameraMatrix=C_mat, distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE)
        
        transform = np.eye(4)
        transform[:3,:3] = cv2.Rodrigues(rvec)[0]
        transform[:3, 3] = tvec[:,0]
        return transform
    
    def _get_transform_PnPRansac(self, c_pnts3d, n_pnts3d, C_mat):
        c_pnts2d, _ = cv2.projectPoints(c_pnts3d, np.zeros(3), np.zeros(3), C_mat, distCoeffs=None)
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(n_pnts3d, c_pnts2d, cameraMatrix=C_mat, distCoeffs=None)
        # Inliers - indices
        
        transform = np.eye(4)
        transform[:3,:3] = cv2.Rodrigues(rvec)[0]
        transform[:3, 3] = tvec[:,0]
        return transform
        
    def _get_transform_LM(self, c_pnts3d, n_pnts3d, C_mat):
        initial = np.zeros(6)
        args = (
            c_pnts_3d,
            n_pnts_3d
        )
        optRes = least_squares(self.estimate_transform_3d, initial, method='lm', max_nfev=10000, args=args, verbose=2)
        x = optRes.x
        
        transform = np.eye(4)
        transform[:3,:3] = R.from_rotvec(x[:3]).as_matrix()
        transform[:3, 3] = x[3:]
        return transform

    def get_transform(self, c_pnts3d, n_pnts3d, C_mat, type_='PnP'):
        types = {
            'PnP': self._get_transform_PnP,
            'PnPRansac': self._get_transform_PnPRansac
        }
        return types[type_](c_pnts3d, n_pnts3d, C_mat)
        
    @staticmethod
    def estimate_transform_2d(x, c_pnts_2d, n_pnts_2d, c_pnts_3d, n_pnts_3d, P_mtrx):
        """
            x - [rotvec, transform]
        """
        transform = get_transform_matrix(x)

        Proj_frwrd = P_mtrx @ np.linalg.inv(transform)
        Proj_bcwrd = P_mtrx @ transform

        n_pred_pnts_2d = vo.reproject_3d_to_2d(c_pnts_3d, Proj_frwrd)
        c_pred_pnts_2d = vo.reproject_3d_to_2d(n_pnts_3d, Proj_bcwrd)

        c_err = c_pnts_2d - c_pred_pnts_2d
        n_err = n_pnts_2d - n_pred_pnts_2d

        residual = np.vstack((c_err*c_err,n_err*n_err))
        return residual.flatten()

    @staticmethod
    def estimate_transform_3d(x, c_pnts_3d, n_pnts_3d):
        """
            x - [rotvec, transform]
        """
        transform = get_transform_matrix(x)

        c_pred_pnts_3d = vo.transform_3d(transform, n_pnts_3d)
        n_pred_pnts_3d = vo.transform_3d(np.linalg.inv(transform), c_pnts_3d)

        c_err = np.linalg.norm(c_pnts_3d - c_pred_pnts_3d, axis=1)
        n_err = np.linalg.norm(n_pnts_3d - n_pred_pnts_3d, axis=1)

        residual = np.vstack((c_err,n_err))
        return residual.flatten()
    
    
def draw_matches(img1, kp1, img2, kp2, matches, color=None): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to 
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (img1.shape[0]+img2.shape[0], max(img1.shape[1], img2.shape[1]), img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (img1.shape[0]+img2.shape[0], max(img1.shape[1], img2.shape[1]))
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[img1.shape[0]:img1.shape[0]+img2.shape[0],0:img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 10
    thickness = 2
    if color:
        c = color
    else:
        # Generate random color for RGB/BGR and grayscale images as needed.
        c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        c = (0, 255, 0)
    for m in matches:
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([0, img1.shape[0]]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    
    return new_img
    
    
def draw_keypoints(c_img, n_img, c_feats, n_feats, radius=5, color=(20, 255, 20)):

    for p in c_feats:
        cv2.circle(c_img, tuple(p), radius, color, 2)
    
    for p in n_feats:
        cv2.circle(n_img, tuple(p), radius, color, 2)

