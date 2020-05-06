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
        
        print('Intrinsics:')
        for name, intr in self.intricsics.items():
            print(f'{name}:\n{intr}')
        print('Extrinsics to cam0:')
        for name, extr in self.cam0_extrinsics.items():
            print(f'{name}:\n{extr}')
        print('Extrinsics to velo:')
        for name, extr in self.velo_extrinsics.items():
            print(f'{name}:\n{extr}')
        print('Baselines:')
        print(self.baselines)

        
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
        p_velo2 = np.linalg.inv(self.velo_extrinsics['T2']).dot(p_cam)
        p_velo3 = np.linalg.inv(self.velo_extrinsics['T3']).dot(p_cam)

        self.baselines = {}
        self.baselines['rgb'] = np.linalg.norm(p_velo3 - p_velo2)
        
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
    
    def _intrinsics_dict(self, mtrx):
        return {
            'cx': mtrx[0,2],
            'cy': mtrx[1,2],
            'fx': mtrx[0,0],
            'fy': mtrx[1,1]
        }
    
    def get_color_intrinsics_dicts(self):
        left = _intrinsics_dict(self.intricsics['K_cam2'])
        right = _intrinsics_dict(self.intricsics['K_cam3'])
        return left, right
    
    def get_color_left_Q_matrix(self):
        # Left
        intr = self._intrinsics_dict(self.intricsics['K_cam2'])
        baseline = self.baselines['rgb']
        
        Q = np.array([
            [1, 0, 0, -intr['cx']],
            [0, 1, 0, -intr['cy']],
            [0, 0, 0, -intr['fx']],
            [0, 0, -1/baseline, 0]
        ])
        
        return Q

    def get_color_images(self, idx):
        fname = self._get_image_fname(idx)
        left_img_fpath = os.path.join(self.LEFT_IMAGES_DIR, fname)
        right_img_fpath = os.path.join(self.RIGHT_IMAGES_DIR, fname)
        l_img = cv2.imread(left_img_fpath)
        l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB)
        r_img = cv2.imread(right_img_fpath)
        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
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
    
    
def VisualOdometry(object):
    def __init__(self):
        pass
    
    def process_depth(self, l_img, r_img, Q):
        left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16*9, 
            blockSize=15,
            P1=0,
            P2=0,
            disp12MaxDiff=1,
        #     preFilterCap=1,
            uniquenessRatio=5,
            speckleWindowSize=200,
            speckleRange=8
        )

        l_img_gray = cv2.cvtColor(l_img, cv2.COLOR_RGB2GRAY)
        r_img_gray = cv2.cvtColor(r_img, cv2.COLOR_RGB2GRAY)

        l_disp = left_matcher.compute(l_img_gray, r_img_gray)        
        disp = l_disp.astype(np.float32) / 16.0
    
        depth = cv2.reprojectImageTo3D(disp, Q)

#         idxs = (disp > 0) & (points[:,:,2] < 100)
#         points = points[idxs]
#         colors = l_img[idxs]
        
        return depth
    
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
    