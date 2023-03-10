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
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from utils import tocuda


# Define torch modules
class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


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
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(
                64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(
                32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(
                16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
            ),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


# probility_volume: [batch_size, D, H, W]
# depth_values: [batch_size, D]
# return: [batch_size, 1, H, W]
def depth_regression(probility_volume, depth_values):
    d = depth_values.view(probility_volume.shape[0], probility_volume.shape[1], 1, 1)
    depth = torch.sum(d * probility_volume, 1)
    return depth


# src_fea: [B, C, H, W]
# src_proj: [B, 4, 4]
# ref_proj: [B, 4, 4]
# depth_values: [B, Ndepth]
# out: [B, C, Ndepth, H, W]
def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():  # What's role of this?
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid(
            [
                torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                torch.arange(0, width, dtype=torch.float32, device=src_fea.device),
            ]
        )
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(
            1, 1, num_depth, 1
        ) * depth_values.view(
            batch, 1, num_depth, 1
        )  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack(
            (proj_x_normalized, proj_y_normalized), dim=3
        )  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(
        src_fea,
        grid.view(batch, num_depth * height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
    )
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


# Define a LightningModule
class MVSVolume(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.FeatureExtractor = FeatureExtractor()
        self.RegularizeVolume = CostRegNet()
        self.training = True

    def build_volume(self, features, proj_matrices, depth_values):
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        num_depth = depth_values.shape[1]
        num_views = len(features)

        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume  # no new space allocated
        volume_sq_sum = ref_volume**2  # square will allocate new space
        del ref_volume  # only delete reference to ref_volume, the data is preserved for volume_sum

        for src_fea, src_proj in zip(src_features, src_projs):
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume**2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(
                    2
                )  # the memory of warped_volume has been modified
            del warped_volume, src_fea, src_proj
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum / (num_views) - (volume_sum / num_views).pow(2)
        return volume_variance

    """
    batch consists of these:
      imgs: [batch_size, nviews, 3, 640, 512]
      intrinsics: [batch_size, nviews, 3, 3]
      extrinsics: [batch_size, nviews, 3, 3]
      depth_gt: [batch_size, 160, 128]
    """
    def training_step(self, batch, batch_idx):
        (
            imgs,
            intrinsics,
            extrinsics,
            proj_matrices,
            depth_gt,
            depth_values,
            mask,
        ) = list(tocuda(batch).values())
        imgs = torch.unbind(
            imgs, 1
        )  # imgs is a tuple of neighbor imgs [batch_size, 3, H, W], length N
        intrinsics = torch.unbind(intrinsics, 1)
        extrinsics = torch.unbind(extrinsics, 1)

        features = [self.FeatureExtractor(i) for i in imgs]
        volume_variance = self.build_volume(features, proj_matrices, depth_values)
        probility_volume = F.softmax(
            self.RegularizeVolume(volume_variance).squeeze(1), dim=1
        )
        depth_map = depth_regression(probility_volume, depth_values).transpose(1, 2)
        loss = F.smooth_l1_loss(
            depth_map[mask], depth_gt[mask], size_average=True
        )  # TODO: select a better loss function

        # Log starts from here
        self.logger.experiment.add_scalar(
            "depth loss", loss, self.global_step
        )  # for consistent with pre 2 epoch
        self.logger.experiment.add_scalar("train/depth loss", loss, self.global_step)
        if self.global_step % 5 == 0:
            thres2mm_error = self.Thres_metrics(depth_map, depth_gt, mask > 0.5, 2)
            thres4mm_error = self.Thres_metrics(depth_map, depth_gt, mask > 0.5, 4)
            thres8mm_error = self.Thres_metrics(depth_map, depth_gt, mask > 0.5, 8)
            self.logger.experiment.add_scalar(
                "train/thres2mm_error", thres2mm_error, self.global_step
            )
            self.logger.experiment.add_scalar(
                "train/thres4mm_error", thres4mm_error, self.global_step
            )
            self.logger.experiment.add_scalar(
                "train/thres8mm_error", thres8mm_error, self.global_step
            )
            self.logger.experiment.add_image(
                "train/reference image", imgs[0][0].transpose(1, 2), self.global_step
            )
            self.logger.experiment.add_image(
                "train/depth_gt", depth_gt[0], self.global_step, dataformats="HW"
            )
            self.logger.experiment.add_image(
                "train/depth_map",
                (depth_map * mask)[0],
                self.global_step,
                dataformats="HW",
            )
        return {
            "loss": loss,
            "depth_map": depth_map,
            "depth_gt": depth_gt,
            "mask": mask,
        }

    def Thres_metrics(self, depth_est, depth_gt, mask, thres):
        assert isinstance(thres, (int, float))
        depth_est, depth_gt = depth_est[mask], depth_gt[mask]
        errors = torch.abs(depth_est - depth_gt)
        err_mask = errors > thres
        return torch.mean(err_mask.float())

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar(
            "average loss per epoch", avg_loss, self.current_epoch
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0
        )
        return optimizer

    def validation_step():
        pass

    def test_step():
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx):
        (
            imgs,
            intrinsics,
            extrinsics,
            proj_matrices,
            depth_gt,
            depth_values,
            mask,
        ) = list(tocuda(batch).values())
        imgs = torch.unbind(
            imgs, 1
        )  # imgs is a tuple of neighbor imgs [batch_size, 3, H, W], length N
        result = self.training_step(batch, batch_idx)
        loss, depth_map, depth_gt, mask = (
            result["loss"],
            result["depth_map"],
            result["depth_gt"],
            result["mask"],
        )
        if batch_idx % 5 == 0:
            print("True")
            self.logger.experiment.add_scalar("test/rdepth loss", loss, batch_idx)
            self.logger.experiment.add_image(
                "test/reference image", imgs[0][0].transpose(1, 2), batch_idx
            )
            self.logger.experiment.add_image(
                "test/depth_gt", depth_gt[0], batch_idx, dataformats="HW"
            )
            self.logger.experiment.add_image(
                "test/depth_map", (depth_map * mask)[0], batch_idx, dataformats="HW"
            )
        return None


parser = argparse.ArgumentParser(description="A PyTorch Implementation of MVSNeRF")
parser.add_argument("--resume", action="store_true", help="resume from lastest model")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a dataset
batch_size = 1
dataset = DTU_dataset(data_dir="./dtu_train/", mode="train", nviews=3, ndepths=256)
train_loader = DataLoader(
    dataset, batch_size, shuffle=True, num_workers=8, drop_last=True
)


model = MVSVolume()
model.to(device)

checkpoint_callback = ModelCheckpoint(every_n_train_steps=3000)
# Train the model
if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        # imgs, intrinsics, extrinsics, ref_depth = dataset[0]
        if args.resume:
            tensorboard = TensorBoardLogger(
                save_dir="./tb_logs/", name="default", version="version_17"
            )
            trainer = pl.Trainer(
                logger=tensorboard,
                checkpoint_callback=checkpoint_callback,
                max_epochs=3,
                gpus=1,
                resume_from_checkpoint="./tb_logs/default/version_17/checkpoints/epoch=1-step=54193.ckpt",
            )
        else:
            tensorboard = TensorBoardLogger(save_dir="./tb_logs/")
            trainer = pl.Trainer(
                logger=tensorboard,
                checkpoint_callback=checkpoint_callback,
                max_epochs=6,
                gpus=1,
            )  # train from scratch
        trainer.fit(model, train_dataloader=train_loader)
