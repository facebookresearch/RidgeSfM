# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted From https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
# MIT License

# Copyright (c) 2019 Duo LI

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch

class h_sigmoid(torch.nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = torch.nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(torch.nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


def conv_4x4_bn(inp, oup, stride):
    return torch.nn.Sequential(
        torch.nn.Conv2d(inp, oup, 4, stride, 1, bias=False),
        torch.nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_3x3_bn(inp, oup, stride):
    return torch.nn.Sequential(
        torch.nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        torch.nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return torch.nn.Sequential(
        torch.nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        torch.nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(torch.nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, pad, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [-2, 1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = torch.nn.Sequential(
                # dw
                (
                    torch.nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size,
                        stride,
                        pad,
                        groups=hidden_dim,
                        bias=False)
                    if stride > 0 else
                    torch.nn.ConvTranspose2d(
                        hidden_dim, hidden_dim, kernel_size, -stride, pad, groups=hidden_dim, bias=False)
                ),
                torch.nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else torch.nn.ReLU(inplace=True),
                # pw-linear
                torch.nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup),
            )
        else:
            self.conv = torch.nn.Sequential(
                # pw
                torch.nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else torch.nn.ReLU(inplace=True),
                # dw
                (
                    torch.nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size,
                        stride,
                        pad,
                        groups=hidden_dim,
                        bias=False)
                    if stride > 0 else
                    torch.nn.ConvTranspose2d(
                        hidden_dim, hidden_dim, kernel_size, -stride, pad, groups=hidden_dim, bias=False)
                ),
                torch.nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else torch.nn.ReLU(inplace=True),
                # pw-linear
                torch.nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
