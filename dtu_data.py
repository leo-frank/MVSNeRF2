import os
import cv2
import torch
import utils
import numpy as np
from torch.utils.data import DataLoader, Dataset

class DTU_dataset(Dataset):
    def __init__(self, data_dir, mode, nviews, ndepths):
        super(DTU_dataset, self).__init__()
        assert mode in ['train', 'validate', 'test']
        # scans_list depends on mode
        with open(os.path.join(os.curdir, '{}_list.txt'.format(mode))) as f:
            self.scans_list = [i.strip() for i in f.readlines()]
        self.depth_dir = os.path.join(data_dir, 'Depths_Raw')
        self.camera_dir = os.path.join(data_dir, 'Cameras')
        self.img_dir = os.path.join(data_dir, 'Rectified')
        self.nviews = nviews
        self.pairs = []
        for scan in self.scans_list:
            with open(os.path.join(self.camera_dir, 'pair.txt')) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    for light_idx in range(7):
                        self.pairs.append((scan, light_idx, ref_view, src_views))
        self.ndepths = ndepths
    
    # output: [3, H, W]
    def read_image(self, filename):
        return torch.Tensor(cv2.imread(filename, cv2.IMREAD_COLOR).transpose(2, 1, 0))
    
    # extrinsics: Tensor [4, 4]
    # intrinsics: Tensor [3, 3]
    # depth_min: float
    # depth_interval: float
    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.strip() for line in f.readlines()]
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])
        return intrinsics, extrinsics, depth_min, depth_interval
    
    # Input: 
    def read_depth(self, filename):
        depth, scale = utils.read_pfm(filename)
        return torch.Tensor(depth.copy())
    
    # imgs: a list of images, each has shape [3, 640, 512]
    # intrinsics: a list of intrinsic, each has shape [3, 3]
    # extrinsics: a list of extrinsic, each has shape [3, 3]
    # depth_gt: ground truth depth map, has shape [160, 128]
    # depth_values: a array of depth hypothesis
    def __getitem__(self, index):
        scan, light_idx, ref_view, src_views = self.pairs[index]
        # print(scan, light_idx, ref_view, src_views)
        view_ids = [ref_view] + src_views[:self.nviews - 1]
        imgs = []
        intrinsics = []
        extrinsics = []
        proj_matrices = []
        for i, id in enumerate(view_ids):
            image_filename = os.path.join(self.img_dir, '{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, id + 1, light_idx))
            imgs.append(self.read_image(image_filename))
            proj_mat_filename = os.path.join(self.camera_dir, 'train/{:0>8}_cam.txt').format(id)
            intrinsic, extrinsic, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
            # multiply intrinsics and extrinsics to get projection matrix
            # print(type(extrinsic))
            proj_mat = extrinsic.copy()
            # print(type(proj_mat))
            proj_mat[:3, :4] = np.matmul(intrinsic, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)


            if i == 0: # For reference image only
                depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval)
                maskname = os.path.join(self.depth_dir, scan+'_train', 'depth_visual_{:0>4}.png'.format(id))
                mask = cv2.imread(maskname, cv2.IMREAD_GRAYSCALE) / 255 
                mask = mask > 0.5
                ref_depth_filename = os.path.join(self.depth_dir, scan+'_train', 'depth_map_{:0>4}.pfm'.format(id))
                ref_depth = self.read_depth(ref_depth_filename)
        # print("imgs[0].shape={}".format(imgs[0].shape))
        return {
            "imgs": np.stack(imgs, 0), # contains 1 ref image and N source images
            "intrinsics": np.stack(intrinsics, 0), # contains 1 ref image and N source images
            "extrinsics": np.stack(extrinsics, 0), # contains 1 ref image and N source images
            "proj_matrices": np.stack(proj_matrices),
            "depth_gt": ref_depth, # only ground truth depth of ref image
            "depth_values": np.array(depth_values), # a array of depth hypothesis
            "mask": mask
        }

    def __len__(self):
        return len(self.pairs)
