# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt, cv2, glob, numpy as np, torch, os, sys
import PIL.Image, torchvision.transforms.functional as TTF, torch.nn.functional as F, imageio
import sklearn.cluster, scipy.interpolate

folder=sys.argv[1]
color_files=[]
depth_files=[]
poses=[]

for fn in sorted(glob.glob(folder+'/pose/*txt'),key=lambda a: int(a.split('/')[-1][:-4])):
    x=int(fn.split('/')[-1].split('.')[0])
    cm=torch.from_numpy(np.loadtxt(fn)).float()
    poses.append(cm)
    color_files.append(folder+'/color/%d.jpg'%x)
    depth_files.append(folder+'/depth/%d.png'%x)

for cf,df in zip(color_files,depth_files):
    #Resize RGB images to 640x480 to match depth images
    im_c=PIL.Image.open(cf)
    im_d=PIL.Image.open(df)
    im_c=im_c.resize(im_d.size)
    im_c.save(cf)

torch.save({
    'intrinsic': np.loadtxt(folder+'/intrinsic/intrinsic_depth.txt'),
    'color': color_files,
    'depth': depth_files,
    'poses': torch.stack(poses),
},folder+'/files.pth')
print(folder)

