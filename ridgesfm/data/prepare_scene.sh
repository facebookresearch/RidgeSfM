# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

echo python SensReader/reader.py --filename scannet_sens/${1}/scene${2}/scene${2}.sens --output_path scannet/${1}/${2} --export_depth_images --export_color_images --export_poses --export_intrinsics
echo python prepare_scene.py `pwd`/scannet/${1}/${2}
