import torch.nn as nn
import torch.nn.functional as F


class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t):
        super(LinearBottleNeck, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class MobileNetV2(nn.Module):
    """
        MobileMetV2 implementation
    """
    def __init__(self, args):
        super(MobileNetV2, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 2, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, args.num_classes, 1)

    def _make_stage(self, n, in_channels, out_channels, stride, t):
        layers = [LinearBottleNeck(in_channels, out_channels, stride, t)]

        while n - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            n -= 1

        return nn.Sequential(*layers)

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:  #
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        # ?
        x = F.adaptive_max_pool2d(x, 1)
        result = {'representation': x.view(x.size(0), -1)}
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        result['output'] = x
        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input.unsqueeze(2).unsqueeze(2)
        z = self.conv2(z)
        z = z.view(z.size(0), -1)
        result = {'output': z}
        if logit:
            result['logit'] = z
        return result
