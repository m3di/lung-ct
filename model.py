from typing import Any

import torch

class ResBlock3D(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=(1,1), padding=(0,0), downsample=None, dropout=0.2):
        super(ResBlock3D, self).__init__()

        self.conv_1 = torch.nn.Conv3d(in_ch, out_ch, kernel_size, stride[0], padding[0])
        self.conv_2 = torch.nn.Conv3d(out_ch, out_ch, kernel_size, stride[1], padding[1])

        self.bn_1 = torch.nn.BatchNorm3d(out_ch)
        self.bn_2 = torch.nn.BatchNorm3d(out_ch)

        self.drop = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.ReLU()

        if downsample != None:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv3d(in_ch, out_ch, downsample[0], downsample[1], downsample[2]),
                torch.nn.BatchNorm3d(out_ch)
            )
        else:
            self.downsample = False

    def forward(self, x):
        residual = x

        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.activation(out)
        out = self.drop(out)
        out = self.conv_2(out)
        out = self.bn_2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)
        out = self.drop(out)

        return out

class ResNet3D(torch.nn.Module):

    def __init__(self):
        super(ResNet3D, self).__init__()

        self.res_blocks = torch.nn.Sequential(
            torch.nn.Conv3d(1, 16, 7, 2, padding=3),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),

            ResBlock3D(16, 16, 3, stride=(1,1), padding=(1, 1), downsample=None),
            ResBlock3D(16, 16, 3, stride=(1,1), padding=(1, 1), downsample=None),
            ResBlock3D(16, 16, 3, stride=(1,1), padding=(1, 1), downsample=None),

            ResBlock3D(16, 32, 3, stride=(2,1), padding=(1, 1), downsample=(3, 2, 1)),

            ResBlock3D(32, 32, 3, stride=(1,1), padding=(1, 1), downsample=None),
            ResBlock3D(32, 32, 3, stride=(1,1), padding=(1, 1), downsample=None),
            ResBlock3D(32, 32, 3, stride=(1,1), padding=(1, 1), downsample=None),

            ResBlock3D(32, 64, 3, stride=(2,1), padding=(1, 1), downsample=(3, 2, 1)),

            ResBlock3D(64, 64, 3, stride=(1,1), padding=(1, 1), downsample=None),
            ResBlock3D(64, 64, 3, stride=(1,1), padding=(1, 1), downsample=None),
            ResBlock3D(64, 64, 3, stride=(1,1), padding=(1, 1), downsample=None),

            ResBlock3D(64, 128, 3, stride=(2,1), padding=(1, 1), downsample=(3, 2, 1)),

            ResBlock3D(128, 128, 3, stride=(1,1), padding=(1, 1), downsample=None),
            ResBlock3D(128, 128, 3, stride=(1,1), padding=(1, 1), downsample=None),
            ResBlock3D(128, 128, 3, stride=(1,1), padding=(1, 1), downsample=None),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.AvgPool3d(8, 8)
        )

        self.out = torch.nn.Linear(128, 2)

    def forward(self, out):
        out = self.res_blocks(out)
        out = self.fc(out)
        out = out.reshape(out.shape[0], -1)
        out = self.out(out)
        return out