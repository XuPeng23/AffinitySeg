import time

import torch
import torch.nn as nn
import numpy as np


class dist_MLP(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.fc1 = nn.Linear(channel, 10)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(10, channel)
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


class DistLoss(nn.Module):
    def __init__(self, curRad=None, alpha=0.5, beta=2):
        super(DistLoss, self).__init__()
        if curRad is None:
            curRad = [1, 2, 5]
        self.curRad = curRad
        self.kernel_size = 2 * self.curRad[-1] + 1
        self.pad_size = int(self.kernel_size / 2)
        self.pad = torch.nn.ReflectionPad2d(self.pad_size)
        self.alpha = alpha
        self.beta = beta
        self.index_map = None
        self.mlp = dist_MLP(25)

    # 获取1，2，5 邻域特征
    def get_feats(self, input_data, curRad=None):
        data = self.pad(input_data)
        data = data.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1).flatten(4)
        center = int(self.kernel_size / 2)
        index_mat = torch.zeros(size=(self.kernel_size, self.kernel_size))
        if curRad is None:
            curRad = self.curRad
        # self.curRad[1, 2, 5]
        for i in curRad:
            index_mat[center - i, center - i] = 1
            index_mat[center - i, center] = 1
            index_mat[center - i, center + i] = 1
            index_mat[center, center - i] = 1
            index_mat[center, center] = 1
            index_mat[center, center + i] = 1
            index_mat[center + i, center - i] = 1
            index_mat[center + i, center] = 1
            index_mat[center + i, center + i] = 1
        index_mat = index_mat.view(-1).numpy()
        index = np.argwhere(index_mat == 1).squeeze()
        data = data[:, :, :, :, index]
        return data.permute(0, 2, 3, 1, 4)

    # 计算距离，中心点与领域的差的绝对值求和
    def compute_dist(self, x):
        feat_neigh_col = self.get_feat_by_index(x)
        size = feat_neigh_col.size(-1)
        feat_center_col = x.permute(0, 2, 3, 1).unsqueeze(-1).repeat(1, 1, 1, 1, size)  # copy到与邻近点数量相同，挨个计算距离 [1,88,88,2,1]
        dist_l1 = torch.abs(feat_center_col - feat_neigh_col)
        dist_l1 = torch.sum(dist_l1, dim=3)  # 压缩
        dist_l1 = self.mlp(dist_l1)
        return dist_l1

    def single_scale_loss(self, x, y):
        dist_l1 = self.compute_dist(x)
        label_neigh_col = self.get_feat_by_index(y)
        size = label_neigh_col.size(-1)
        label_center_col = y.permute(0, 2, 3, 1).unsqueeze(-1).repeat(1, 1, 1, 1, size)
        label_l1 = (label_neigh_col == label_center_col).squeeze(dim=3).float()
        mask = label_l1 == 0
        label_l1[mask] = -1
        dist_loss = dist_l1 * label_l1
        dist_loss[mask] = dist_loss[mask] + self.beta
        dist_loss[~mask] = dist_loss[~mask] - self.alpha
        dist_loss = torch.max(dist_loss, torch.zeros_like(dist_loss))
        dist_loss[mask] = dist_loss[mask] * (~mask).sum()
        dist_loss[~mask] = dist_loss[~mask] * mask.sum()
        return dist_l1, dist_loss.sum() / (
                dist_loss.numel() ** 2)

    def get_feat_by_index(self, x, index=None):  # x:[1,2,88,88]
        B, C, H, W = x.shape
        if self.index_map is None or self.index_map.size(0) != H * W:
            self.index_map = self.get_index(H, W)
        if index is None:
            index = self.index_map
        x = x.flatten(-2)  # [1,2,7744]
        feat = x[:, :, self.index_map].view(B, C, H, W, -1)  # [1,2,88,88,1]
        return feat.permute(0, 2, 3, 1, 4)  # [1,88,88,2,1]

    def get_index(self, H, W):
        b = [torch.tensor([[-r, 0, r]]) * self.kernel_size + torch.tensor([[-r, 0, r]]).t() for r in self.curRad]
        b = torch.cat(b, dim=1)
        c = b.flatten() + (self.kernel_size * self.kernel_size - 1) / 2
        c = c.numpy()
        # 删除重复元素即删除多余的两个中心点
        c = np.unique(c)
        c = np.sort(c)
        index_map = torch.arange(H * W).reshape(H, W).unsqueeze(0).unsqueeze(0).to(dtype=torch.float)
        index_map = self.pad(index_map).squeeze()
        # 88 x 88 x 11 x 11
        index_map = index_map.unfold(0, self.kernel_size, 1).unfold(1, self.kernel_size, 1).flatten(0, 1).flatten(-2)
        index_map = index_map[:, c].to(torch.long)
        return index_map

    def compute_distv1(self, x):
        feat_neigh_col = self.get_feats(x)
        size = feat_neigh_col.size(-1)
        feat_center_col = x.permute(0, 2, 3, 1).unsqueeze(-1).repeat(1, 1, 1, 1, size)
        dist_l1 = torch.abs(feat_center_col - feat_neigh_col)
        dist_l1 = torch.sum(dist_l1, dim=3)
        return dist_l1

    def single_scale_lossv1(self, x, y):
        dist_l1 = self.compute_distv1(x)
        label_neigh_col = self.get_feats(y)
        size = label_neigh_col.size(-1)
        label_center_col = y.permute(0, 2, 3, 1).unsqueeze(-1).repeat(1, 1, 1, 1, size)
        label_l1 = (label_neigh_col == label_center_col).squeeze(dim=3).float()
        mask = label_l1 == 0
        label_l1[mask] = -1
        dist_loss = dist_l1 * label_l1
        dist_loss[mask] = dist_loss[mask] + self.beta
        dist_loss[~mask] = dist_loss[~mask] - self.alpha
        dist_loss = torch.max(dist_loss, torch.zeros_like(dist_loss))
        dist_loss[mask] = dist_loss[mask] * (~mask).sum()
        dist_loss[~mask] = dist_loss[~mask] * mask.sum()
        return dist_l1, dist_loss.sum() / (
                dist_loss.numel() ** 2)

    def forward(self, x, y):
        dist_mat, loss = self.single_scale_loss(x, y)
        return dist_mat, loss


if __name__ == '__main__':
    feat = torch.randn(1, 2, 88, 88)
    # feat = feat.unfold(2, 11, 1).unfold(3, 11, 1).flatten(4)
    # print(feat.shape)
    torch.set_printoptions(precision=1)
    # np.savetxt('123.txt', feat.squeeze().numpy(), fmt='%.1e')
    # print(feat.squeeze())
    mask = torch.randn(1, 1, 88, 88)
    mask = torch.where(mask > 0, 1., 0.)
    loss_func = DistLoss(curRad=[1, 2, 5])
    f = loss_func.get_feat_by_index(feat)
    # start = time.time()
    # dist, loss = loss_func.single_scale_loss(feat, mask)
    # distv, lossv = loss_func.single_scale_lossv1(feat, mask)
    # print(time.time() - start)
    # print(loss)
    # print(lossv)
    # print(loss_func.get_feats(feat).shape)
    # print(loss_func.get_feat_by_index(feat).shape)
    # m = loss_func.get_feats(feat) == loss_func.get_feat_by_index(feat)
    # print(m.sum())
    # loss_func.padTest(feat)
    # f = loss_func.get_feats(feat)
    # print(f.shape)
    # mats, loss = loss_func(feat, mask)
    # print(mats.shape)
    # print(loss)
    # fx = torch.randn(11, 11)
    # print(fx)
    # print(fx[feat[:, 0], [feat[:, 1]]])
