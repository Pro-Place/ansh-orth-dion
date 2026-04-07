"""
Wide ResNet models for CIFAR-10/CIFAR-100 scaling experiments.

WideResNet-d-k: d = depth, k = width multiplier.
Configurations: WRN-28-2, WRN-28-4, WRN-28-10, WRN-40-4.

Reference: Zagoruyko & Komodakis, "Wide Residual Networks", BMVC 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, depth=28, width_factor=10, num_classes=10, dropout_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth must be 6n+4"
        n = (depth - 4) // 6
        widths = [16, 16 * width_factor, 32 * width_factor, 64 * width_factor]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, widths[0], 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(widths[1], n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(widths[2], n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(widths[3], n, stride=2, dropout_rate=dropout_rate)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layer(self, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(WideBasicBlock(self.in_planes, planes, s, dropout_rate))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def create_wide_resnet(depth=28, width=10, num_classes=10, dropout=0.0):
    return WideResNet(depth, width, num_classes, dropout)


# Model configs for scaling experiment
WIDE_RESNET_CONFIGS = {
    "wrn-28-2":  {"depth": 28, "width": 2,  "params_approx": "1.5M"},
    "wrn-28-4":  {"depth": 28, "width": 4,  "params_approx": "5.9M"},
    "wrn-28-10": {"depth": 28, "width": 10, "params_approx": "36.5M"},
    "wrn-40-4":  {"depth": 40, "width": 4,  "params_approx": "8.9M"},
}

if __name__ == "__main__":
    for name, cfg in WIDE_RESNET_CONFIGS.items():
        m = create_wide_resnet(cfg["depth"], cfg["width"])
        params = sum(p.numel() for p in m.parameters())
        matrix_params = sum(p.numel() for p in m.parameters() if p.ndim >= 2)
        x = torch.randn(2, 3, 32, 32)
        y = m(x)
        print(f"{name:12s} | params: {params:>10,} | matrix: {matrix_params:>10,} "
              f"({100*matrix_params/params:.1f}%) | output: {y.shape}")
