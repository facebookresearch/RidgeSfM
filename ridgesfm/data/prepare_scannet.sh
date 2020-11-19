# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

mkdir scannet
mkdir scannet/train
mkdir scannet/test
parallel < scenes.sh
mkdir scannet/validation_scenes
mkdir scannet/test_scenes
python generate_scannet_test.py
