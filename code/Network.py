import torch
import torch.nn as nn
from resnext import ResNeXt101
import torch.nn.functional as F
from Dist_Embed_MLP import DistLoss


class DistModule(nn.Module):
    def __init__(self, dim=None, hidden_dim=None):
        super(DistModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=hidden_dim, kernel_size=1)
        self.out = nn.Conv2d(in_channels=dim * 3, out_channels=dim, kernel_size=1)
        self.loss_func = DistLoss(alpha=0.5, beta=2.).cuda()
        self.r = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.BN = nn.BatchNorm2d(hidden_dim)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x, gts=None):
        if gts is not None:

            feat1 = self.conv1(x)

            gt = F.interpolate(gts, size=(feat1.size(2), feat1.size(3)), mode='bilinear',align_corners = True)
            gt = gt > 0.9
            dist_l1, losses = self.loss_func(feat1, gt)  # 三元损失约束，label一致性做约束

            # m = torch.exp(-self.r * dist_l1) # e的函数把距离变成相关性，学习来的相关性
            m = -self.r * dist_l1
            m = m.softmax(dim=-1).unsqueeze(-1)

            feat1 = self.loss_func.get_feat_by_index(feat1)
            out = feat1 @ m

            out = out.squeeze(-1).permute(0, 3, 1, 2)
            out = out + x
            out = self.BN(out)
            out = self.leaky_relu(out)

            return losses, out

        else:
            feat1 = self.conv1(x)
            # feat2 = self.conv2(x)   原来这里feat忘记换掉了
            dist_l1 = self.loss_func.compute_dist(feat1)
            m = -self.r * dist_l1
            # m = torch.exp(-self.r * dist_l1)
            m = m.softmax(dim=-1).unsqueeze(-1)
            feat1 = self.loss_func.get_feat_by_index(feat1)  # 然后这里也是feat2
            out = feat1 @ m  # 然后这里也是feat2
            out = out.squeeze(-1).permute(0, 3, 1, 2)
            out = out + x
            out = self.BN(out)
            out = self.leaky_relu(out)

            return out


class convA(nn.Module):

    def __init__(self, inch, outch):
        super(convA, self).__init__()

        self.conv2 = nn.Conv2d(inch, outch, (3, 3), padding=1)
        self.batch2 = nn.BatchNorm2d(outch)
        self.relu2 = nn.LeakyReLU(inplace=True)

        self.conv3 = nn.Conv2d(inch, outch, (1, 1), padding=0)
        self.batch3 = nn.BatchNorm2d(outch)
        self.relu3 = nn.LeakyReLU(inplace=True)

        self.conv4 = nn.Conv2d(inch, outch, (5, 5), padding=2)
        self.batch4 = nn.BatchNorm2d(outch)
        self.relu4 = nn.LeakyReLU(inplace=True)

        self.conv7 = nn.Conv2d(inch, outch, (7, 7), padding=3)
        self.batch7 = nn.BatchNorm2d(outch)
        self.relu7 = nn.LeakyReLU(inplace=True)

        self.conv5 = nn.Conv2d(3 * outch, outch, 3, padding=1)
        self.conv6 = nn.BatchNorm2d(outch)
        self.relu6 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        c2 = self.conv2(x)
        b2 = self.batch2(c2)
        r2 = self.relu2(b2)
        c3 = self.conv3(x)
        b3 = self.batch3(c3)
        r3 = self.relu3(b3)
        c4 = self.conv4(x)
        b4 = self.batch4(c4)
        r4 = self.relu4(b4)

        merge = torch.cat([r2, r3, r4], dim=1)
        c5 = self.conv5(merge)
        out1 = self.conv6(c5)
        out = self.relu6(out1)
        return out


class convZ(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(convZ, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        resnext = ResNeXt101()

        self.layer0 = resnext.layer0  # 64  128*128
        self.layer1 = resnext.layer1  # 256   64*64
        self.layer2 = resnext.layer2  # 512   32*32
        self.layer3 = resnext.layer3  # 1024   16*16

        self.maxPool = nn.MaxPool2d(2, return_indices=True)  # 64 128*128    ......

        self.conv1 = convA(3, 64)  # 64 256*256
        self.pool1 = nn.MaxPool2d(2)  # 64 128*128    .........

        self.conv2 = convA(64, 128)  # 128 128*128
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = convA(128, 256)  # 256 64*64    .........
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = convA(256, 512)  # 512 32*32      ........
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = convA(512, 1024)  # 1024 16*16    .........

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  # 512 32*32

        self.conv6 = nn.Conv2d(1024, 512, 3, padding=1)  # 512 32*32
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)  # 256  64*64
        self.conv7 = nn.Conv2d(512, 256, 3, padding=1)  # 256  64*64
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 128  128*128
        self.conv8 = nn.Conv2d(256, 128, 3, padding=1)  # 128  128*128

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 64   256*256
        self.conv9 = convZ(128, 64)  # 64 256*256
        self.conv10 = nn.Conv2d(64, 32, 3, padding=1)  # 32 256*256
        self.conv11 = nn.Conv2d(32, 1, 1)  # 1 256*256

        self.convx1 = nn.Conv2d(128, 64, 3, padding=1)  # 64  128*128
        self.convx2 = nn.Conv2d(512, 256, 3, padding=1)  # 256  64*64
        self.convx3 = nn.Conv2d(1024, 512, 3, padding=1)  # 512   32*32
        self.convx4 = nn.Conv2d(2048, 1024, 3, padding=1)  # 1024  16*16
        self.convxout = nn.Conv2d(3 + 64 + 128, 64, kernel_size=(1, 1), stride=1, padding=0)
        # 模块循环次数
        layers6 = nn.ModuleList([
            DistModule(dim=512, hidden_dim=512) for _ in range(0, 1)
        ])
        self.layers6 = nn.Sequential(*layers6)

        layers7 = nn.ModuleList([
            DistModule(dim=256, hidden_dim=256) for _ in range(0, 1)
        ])
        self.layers7 = nn.Sequential(*layers7)

        layers8 = nn.ModuleList([
            DistModule(dim=128, hidden_dim=128) for _ in range(0, 1)
        ])
        self.layers8 = nn.Sequential(*layers8)

        layers9 = nn.ModuleList([
            DistModule(dim=64, hidden_dim=64) for _ in range(0, 1)
        ])
        self.layers9 = nn.Sequential(*layers9)

        layers10 = nn.ModuleList([
            DistModule(dim=32, hidden_dim=32) for _ in range(0, 1)
        ])
        self.layers10 = nn.Sequential(*layers10)

        layers11 = nn.ModuleList([
            DistModule(dim=1, hidden_dim=1) for _ in range(0, 1)
        ])
        self.layers11 = nn.Sequential(*layers11)

    def forward(self, x, gts=None):
        layer0 = self.layer0(x)  # 64    128
        layer1 = self.layer1(layer0)  # 256   64
        layer2 = self.layer2(layer1)  # 512   32
        layer3 = self.layer3(layer2)  # 1024  16

        x0 = self.conv1(x)  # ////////////////// 64 256*256

        p1, ind1 = self.maxPool(x0)  # 64 128*128    .........
        t1 = torch.cat([p1, layer0], dim=1)  # //////////////////
        x1 = self.convx1(t1)  # 64  128*128
        x1 = self.conv2(x1)  # ////////////////// #128 128*128

        p2, ind2 = self.maxPool(x1)
        c3 = self.conv3(p2)  # 256 64*64    ......... #////////////////// 256 64*64
        t2 = torch.cat([c3, layer1], dim=1)  # //////////////////
        x2 = self.convx2(t2)  ##256  64*64

        p3, ind3 = self.maxPool(x2)
        c4 = self.conv4(p3)  # 512 32*32
        t3 = torch.cat([c4, layer2], dim=1)  # //////////////////
        x3 = self.convx3(t3)  ##512   32*32

        p4, ind4 = self.maxPool(x3)
        c5 = self.conv5(p4)  # 1024 16*16    .........
        t4 = torch.cat([c5, layer3], dim=1)  # //////////////////
        x4 = self.convx4(t4)  # 1024  16*16

        up_6 = self.up6(x4)
        merge6 = torch.cat([up_6, x3], dim=1)
        c6 = self.conv6(merge6)
        # DistModule特征变换
        dist_loss = 0
        if gts is not None:
            for layer in self.layers6:
                losses, c6 = layer(c6, gts)
                dist_loss += losses
        else:
            for layer in self.layers6:
                c6 = layer(c6)

        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, x2], dim=1)
        c7 = self.conv7(merge7)
        if gts is not None:
            for layer in self.layers7:
                losses, c7 = layer(c7, gts)
                dist_loss += losses
        else:
            for layer in self.layers7:
                c7 = layer(c7)

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, x1], dim=1)
        c8 = self.conv8(merge8)
        if gts is not None:
            for layer in self.layers8:
                losses, c8 = layer(c8, gts)
                dist_loss += losses
        else:
            for layer in self.layers8:
                c8 = layer(c8)

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, x0], dim=1)
        c9 = self.conv9(merge9)

        c10 = self.conv10(c9)

        c11 = self.conv11(c10)

        out = nn.Sigmoid()(c11)

        return dist_loss, out