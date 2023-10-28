import torch
from torch import nn

# GCGM Global compersation guide module
class GCGM(nn.Module):
    def __init__(self, in_channels=256, patch_size=2, channels=64):
        super(GCGM, self).__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size),
                                stride=(self.patch_size, self.patch_size))
        # self.flod = nn.Fold(kernel_size=(self.patch_size, self.patch_size),stride=(self.patch_size, self.patch_size), output_size=(channels,channels))
        self.resolution_trans = nn.Sequential(
            nn.Linear(self.patch_size * self.patch_size, 2 * self.patch_size * self.patch_size, bias=False),
            nn.Linear(2 * self.patch_size * self.patch_size, self.patch_size * self.patch_size, bias=False),
            nn.ReLU()
        )
        self.resolution_trans1 = nn.Sequential(
            nn.Linear(self.patch_size * self.patch_size, 2 * self.patch_size * self.patch_size, bias=False),
            nn.Linear(2 * self.patch_size * self.patch_size, self.patch_size * self.patch_size, bias=False),
            nn.ReLU()
        )
        self.resolution_trans2 = nn.Sequential(
            nn.Linear(self.patch_size * self.patch_size, 2 * self.patch_size * self.patch_size, bias=False),
            nn.Linear(2 * self.patch_size * self.patch_size, self.patch_size * self.patch_size, bias=False),
            nn.ReLU()
        )
        self.resolution_trans3 = nn.Sequential(
            nn.Linear(self.patch_size * self.patch_size, 2 * self.patch_size * self.patch_size, bias=False),
            nn.Linear(2 * self.patch_size * self.patch_size, self.patch_size * self.patch_size, bias=False),
            nn.ReLU()
        )
        self.conv_x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_x2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_y = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x1, x2, x3):
        y_sum = x3
        x1_size = list(x1.size())

        x_new_1 = self.conv_x1(x1)
        x_new_2 = self.conv_x2(x2)
        y_sum_1 = self.conv_y(y_sum)

        features = []
        attentions = []
        for i in range(x_new_1.size()[1]):

            # 处理 x1 unfold_x1.size torch.Size([1, 1024, 4])
            unfold_x1 = self.unfold(x_new_1[:, i:i + 1, :, :]).transpose(-1, -2)
            unfold_x1 = self.resolution_trans(unfold_x1).unsqueeze_(-1)# 1,1024,4,1
            unfold_x1 = unfold_x1.transpose(0, 1) # 1024,1,4,1
            # 处理 x2 unfold_x2.size torch.Size([1, 4, 1024])
            unfold_x2 = self.unfold(x_new_2[:, i:i + 1, :, :])
            # unfold_x2 = self.resolution_trans1(unfold_x2.transpose(-1, -2)).transpose(-1, -2)
            unfold_x2 = self.resolution_trans1(unfold_x2.transpose(-1, -2)) # 1,1024,4
            unfold_x2 = unfold_x2.unsqueeze_(-1)  # 1,1024,4,1
            unfold_x2 = unfold_x2.transpose(0, 1)  # 1024,1,4,1

            # att = torch.matmul(unfold_x1, unfold_x2)

            # 处理 y_sum -------------unfold_y------------- torch.Size([1, 4, 1024])
            unfold_y = self.unfold(y_sum_1[:, i:i + 1, :, :])
            # unfold_y = self.resolution_trans2(unfold_y.transpose(-1, -2)).transpose(-1, -2)
            unfold_y = self.resolution_trans2(unfold_y.transpose(-1, -2)) # 1,1024,4
            unfold_y = unfold_y.unsqueeze_(-1)  # 1,1024,4,1
            unfold_y = unfold_y.transpose(0, 1)  # 1024,1,4,1


            t1 = unfold_x1
            t2 = unfold_x2.transpose(-1, -2)
            t3 = unfold_y
            t12 = torch.matmul(t1,t2).sum(dim=-1).unsqueeze_(-2)# 1,1,4,4
            t123 = torch.matmul(t3,t12).sum(dim=-1).transpose(0, 1).transpose(1, 2)

            # att = torch.matmul(unfold_y, att)

            zi = torch.nn.functional.fold(t123, (x1_size[2], x1_size[3]), (self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
            attentions.append(zi)

        attentions = torch.cat((attentions), dim=1)
        return attentions

class CGR(nn.Module):
    def __init__(self, low, high):
        super(CGR, self).__init__()
        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(high, low, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(low),
            nn.ReLU(True))
        self.up_2 = nn.Sequential(
            nn.Conv2d(low, low, kernel_size=1),
            nn.InstanceNorm2d(low),
            nn.ReLU(True))
        self.conv_1 = nn.Sequential(
            nn.Conv2d(low * 2, low, kernel_size=1),
            nn.InstanceNorm2d(low),
            nn.ReLU(True))
        self.conv_3 = nn.Conv2d(low, 8 , kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv_5 = nn.Conv2d(low, 8, kernel_size=5, padding=2, dilation=1, stride=1)
        self.conv_7 = nn.Conv2d(low, 8, kernel_size=7, padding=3, dilation=1, stride=1)
        self.conv_1_1 = nn.Conv2d(24, 1, kernel_size=1)
        self.In = nn.InstanceNorm2d(1)
        self.Re = nn.ReLU(True)



    def forward(self, x1, x2):
        y1 = self.up_1(x1)
        y2 = self.up_2(x2)
        feature12 = torch.cat([y1, y2], dim= 1)
        feature = self.conv_1(feature12)
        feature_3 = self.conv_3(feature)
        feature_5 = self.conv_5(feature)
        feature_7 = self.conv_7(feature)
        feature357 = torch.cat([feature_3, feature_5],dim=1)
        feature357 = torch.cat([feature357, feature_7], dim=1)
        p_f = self.conv_1_1(feature357)
        p_f = self.In(p_f)
        p_f = self.Re(p_f)
        return feature * p_f
if __name__ == '__main__':
    test = GCGM(256).cuda()
    a = torch.randn(1, 256, 64, 64).cuda()
    b = torch.randn(1, 256, 64, 64).cuda()
    d = torch.randn(1, 256, 64, 64).cuda()
    c = test(a, b, d)