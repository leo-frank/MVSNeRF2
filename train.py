import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import cv2
from dtu_data import DTU_dataset
import copy
# Define torch modules
class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # NOTICE: I wanna keep feature map as the same size with input
#         self.encoder = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=4, padding=1)
#     def forward(self, image):
#         # argument 'image' contains only 1 image
#         return self.encoder(image)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x
class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)
class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)
        # 这里的一个细节，是，先降低channel（32->8），然后再不断升高channel（8->64）
        # 原文里说，这事一种类似Unet的结构（encoder-decoder structure）

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        # print(x.shape)
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


# class CostRegNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.reg = nn.Conv3d(32, 1, 3, padding=1)
#     def forward(self, cost_volume):
#         return self.reg(cost_volume)

# Input:
# features: a list of feature; 
# features[0]: Tensor [batch_size, channels, H, W]
# intrinsic: Tensor [batch_size, 3, 3]
# extrinsic: Tensor [batch_size, 4, 4]
# depth_values: Tensor [batch_size, N], each sample is a list of valid depths; depth_values[0][0]: int
# Output:
# cost_volume: [batch_size, D, channels, H, W]
def homography_mapping(features, extrinsics, intrinsics, depth_values, nviews):
    # print("depth_values: {}".format(depth_values))
    # print("depth_values.shape: {}".format(depth_values.shape))
    ref_feature, src_features = features[0], features[1:]
    ref_proj, src_projs = extrinsics[0], extrinsics[1:]
    ref_intrinsic, src_intrinsics = intrinsics[0], intrinsics[1:]
    ref_rotation, ref_trans = ref_proj[:, :3, :3], ref_proj[:, :3, 3]
    batch_size, channels_num, H, W = ref_feature.shape[0], ref_feature.shape[1], ref_feature.shape[2], ref_feature.shape[3]
    # FIXME: ref volume
    # volume = torch.empty([batch_size, depth_values.shape[1], channels_num, H, W]).to(ref_feature.device)
    volume_sum = torch.empty([batch_size, depth_values.shape[1], channels_num, H, W]).to(ref_feature.device)
    volume_square = torch.empty([batch_size, depth_values.shape[1], channels_num, H, W]).to(ref_feature.device)

    
    for src_fea, src_proj, src_intrinsic in zip(src_features, src_projs, src_intrinsics):
        # src_rotation: [batch_size, 3, 3], src_trans: [batch_size, 3, 1]
        src_rotation, src_trans = src_proj[:, :3, :3], src_proj[:, :3, 3]
        relative_rotation = torch.matmul(torch.inverse(src_rotation), ref_rotation)
        relative_trans = ref_trans - src_trans
        normal_T = torch.transpose(torch.Tensor([[0, 0, 1]]), 0, 1).unsqueeze(0).to(ref_feature.device)
        # print("relative_rotation.shape: {}".format(relative_rotation.shape))
        # print("relative_rotation.device: {}".format(relative_rotation.device))
        # print("normal_T.device: {}".format(normal_T.device))
        # print("volume.device: {}".format(volume.device))
        # with torch.no_grad(): # TODO: I am not sure about this
        v = []
        for depth in depth_values[0]:
            Homo = torch.matmul(torch.matmul(src_intrinsic, relative_rotation + torch.transpose(relative_rotation, 1, 2) * relative_trans * normal_T / depth), torch.inverse(ref_intrinsic))
            # print("Homo", Homo.shape) # the shape of Homo should be [3, 3], freedom of 8
            y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=ref_feature.device),
                                    torch.arange(0, W, dtype=torch.float32, device=ref_feature.device)])
            y, x = y.contiguous(), x.contiguous()
            y, x = y.view(H * W), x.view(H * W)
            xyz = torch.stack((x, y, torch.ones_like(x))) # torch.Size([3, 327680])
            proj_xyz = torch.matmul(torch.inverse(Homo), xyz) # [batch_size, 3, H*W]
            proj_xy = proj_xyz[:, :2] / proj_xyz[:, 2]
            proj_x_normalized = proj_xy[:, 0, :] / ((W - 1) / 2) - 1 # [batch_size, H*W]
            proj_y_normalized = proj_xy[:, 1, :] / ((H - 1) / 2) - 1 # [batch_size, H*W]
            grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=2) # [batch_size, H*W, 2]
            grid = grid.view(batch_size, H, W, 2)
            output = F.grid_sample(src_fea, grid)
            # print("output.shape: {}", output.shape) # [1, 8, 160, 128])
            v.append(output)
        volume = torch.stack(v, 1) # .unsqueeze(0)
        volume_sum = volume_sum + volume
        volume_square = volume_square + volume.pow(2)
    cost_volume = volume_square / nviews - (volume_sum / nviews).pow(2)
    return cost_volume.transpose(1,2)

# probility_volume: [batch_size, D, H, W]
# depth_values: [batch_size, D]
def depth_regression(probility_volume, depth_values):
    # print(probility_volume.shape)
    # print(depth_values.shape)
    # print(probility_volume.device)
    # print(depth_values.device)
    # TODO: I am not sure about this
    d = depth_values.view(probility_volume.shape[0], probility_volume.shape[1], 1, 1)
    depth = torch.sum(d * probility_volume, 1)
    return depth

# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper

@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))

def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W] TODO: [batch??, channel, height, width]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad(): # What's role of this?
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        # // 将图像划分为单元网格
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

# Define a LightningModule
class MVSVolume(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.FeatureExtractor = FeatureExtractor()
        self.VolumeReg = CostRegNet()
        self.training = True

    # batch consists of these:
    #   imgs: [batch_size, nviews, 3, 640, 512]
    #   intrinsics: [batch_size, nviews, 3, 3]
    #   extrinsics: [batch_size, nviews, 3, 3]
    #   depth_gt: [batch_size, 160, 128]
    def training_step(self, batch, batch_idx):
        # print(batch)
        imgs, intrinsics, extrinsics, proj_matrices, depth_gt, depth_values, mask= list(tocuda(batch).values())
        torch.save(depth_values, "depth_values.pt")
        imgs = torch.unbind(imgs, 1) # imgs is a tuple of neighbor imgs [batch_size, 3, H, W], length N
        intrinsics = torch.unbind(intrinsics, 1)
        extrinsics = torch.unbind(extrinsics, 1)
        features = [self.FeatureExtractor(i) for i in imgs]
        nviews = len(imgs)
        print("self.training: {}".format(self.training))

        # # step 2. differentiable homograph, build cost volume
        # volume_variance = homography_mapping(features, extrinsics, intrinsics, depth_values, nviews)
        
        # step 2. differentiable homograph, build cost volume
        # print("step 2. differentiable homograph, build cost volume")
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume # TODO: will this allocate an new space in GPU?
        volume_sq_sum = ref_volume ** 2 # square
        del ref_volume # TODO: how to supervise GPU memory changes?
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
            # if self.training:
            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + warped_volume ** 2
            # else:
            #     # TODO: this is only a temporal solution to save memory, better way?
            #     volume_sum += warped_volume
            #     volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        # TODO: error here: inplace operation: 
        # volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        volume_variance = volume_sq_sum.div(num_views).sub(volume_sum.div(num_views).pow(2))
        # print("volume_variance.shape: {}".format(volume_variance.shape)) # ([1, 8, 48, 160, 128])
        
        # print("cost_volume.size: {}".format(cost_volume.shape)) # torch.Size([1, 48, 48, 8, 160, 128])
        # 
        probility_volume = F.softmax(self.VolumeReg(volume_variance).squeeze(1), dim=1)
        torch.save(probility_volume, "probility_volume.pt")
        depth_map = depth_regression(probility_volume, depth_values).transpose(1, 2)
        # print(depth_map.shape)
        # print(depth_gt.shape)
        loss = F.smooth_l1_loss(depth_map[mask], depth_gt[mask]) # TODO: what loss function is better ?
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    def validation_step():
        pass
    def test_step():
        pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a dataset
batch_size = 1
dataset = DTU_dataset(data_dir="./dtu_train/", mode='train', nviews=3)
train_loader = DataLoader(dataset, batch_size, shuffle=True)
print(len(dataset))

# model
model = MVSVolume()

model.to(device)
 
# Train the model
if __name__ == '__main__':
    with torch.autograd.set_detect_anomaly(True):
        # imgs, intrinsics, extrinsics, ref_depth = dataset[0]
        trainer = pl.Trainer(max_epochs=1, gpus=1)
        trainer.fit(model, train_dataloader=train_loader)