# RidgeSfM: Structure from Motion via Robust Pairwise Matching Under Depth Uncertainty
_Benjamin Graham, David Novotny_<br/>
_3DV 2020_

This is the official implementation of **RidgeSfM: Structure from Motion via Robust Pairwise Matching Under Depth Uncertainty** in PyTorch.

[Link to paper](ridge_sfm.pdf)

<a href="output/scannet/" rel="some text"><img src="output/scannet/0708/scene1_frameskip3.gif" width="320" alt="ScanNet reconstruction" /></a>



## RidgeSfM applied to the ScanNet test set


<table>
	<tr>
		<td>

[**Scene 0707_00 frame skip rate _k=1_**](output/scannet/0707/README.md)  <br/>
<a href="output/scannet/0707/README.md"><img src="output/scannet/0707/scene0_0.png" width="240" alt="ScanNet reconstruction" /></a>

  </td>
  <td>

[**Scene 0708_00 frame skip rate _k=3_**](output/scannet/0708/README.md)  <br/>
<a href="output/scannet/0708/README.md"><img src="output/scannet/0708/scene1_0.png" width="240" alt="ScanNet reconstruction" /></a>

</td>
	</tr>
	<tr>
		<td>

[**Scene 0709_00 frame skip rate _k=10_**](output/scannet/0709/README.md)  <br/>
<a href="output/scannet/0709/README.md"><img src="output/scannet/0709/scene2_0.png" width="240" alt="ScanNet reconstruction" /></a>

   </td>
   <td>

[**Scene 0710_00 frame skip rate _k=30_**](output/scannet/0710/README.md)  <br/>
<a href="output/scannet/0710/README.md"><img src="output/scannet/0710/scene3_0.png" width="240" alt="ScanNet reconstruction" /></a>

  </td>
	</tr>
</table>

Below we illustrate the depth uncertainty factors of variation for a frame from scene 0708.<br/>
<table><tr><td>
<img src="output/scannet/0708/fov12a.png" width="720" alt="ScanNet Depth Factors of variation" /><br/>
Top left: an input image. <br/>
Bottom left: the predicted depth. <br/>
Middle and right: We use SVD to reduce the 32 FoV planes down to 12 planes, and display them as 4 RGB images; each of the 4x3 color planes represents one factor of variation.
</td></tr></table>


## RidgeSfM applied to a video taken on a mobile phone

We applied RidgeSfM to a short video taken using a mobile phone camera.
There is no ground truth pose, so the bottom right hand corner of the video is blank.

<table>
	<tr>
		<td>

[**Living room - skip rate _k=3_**](output/cubot/scene0/README.md)  <br/>
<a href="output/cubot/scene0/README.md"><img src="output/cubot/scene0/scene0_0.png" width="240" alt="ScanNet reconstruction" /></a>

</td>
</tr>
</table>

## RidgeSfM applied to the KITTI odometry dataset

We trained a depth prediction network on the [KITTI depth prediction](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) training set. We then processed videos from the [KITTI Visual Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). We used the 'camera 2' image sequences, cropping the input to RGB images of size 1216x320\. We used R2D2 as the keypoint detector. We used a frame skip rate of _k_=3\. The scenes are larger spatially, so for visualization we increased the number of K-Means centroids to one million.

<table>
	<tr>
		<td>

[**Scene 6 - skip rate _k=3_**](output/kitti/scene6/README.md)  <br/>
<a href="output/kitti/scene6/" rel="some text"><img src="output/kitti/scene6/scene6.png" width="240" alt="ScanNet reconstruction" /></a>

  </td>
  <td>

[**Scene 7 - skip rate _k=3_**](output/kitti/scene7/README.md)  <br/>
<a href="output/kitti/scene7/" rel="some text"><img src="output/kitti/scene7/scene7.png" width="240" alt="ScanNet reconstruction" /></a>

  </td>
  </tr>
</table>

# Setup

- Download the [ScanNet dataset](http://www.scan-net.org/) to `ridgesfm/data/scannet_sens/[train|test]/`
- Download [SensReader](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python) to `ridgesfm/data`
- Download [SuperPoint](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.pdf) to ridgesfm/
```
wget https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/weights/superpoint_v1.pth?raw=true -O ridgesfm/weights/superpoint_v1.pth
wget https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/superpoint.py -O ridgesfm/superpoint.py
```
- Run `bash prepare_scannet.sh` in `ridgesfm/data/`
- Run `python ridgesfm.py`

## Dependencies:
- Python 3.7+
- PyTorch 1.5+ and TorchVision
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
- [Faiss](https://github.com/facebookresearch/faiss)
- [PyKeOps](https://pypi.org/project/pykeops/)
- OpenCV
- NumPy


## License
RidgeSfM is CC-BY-NC licensed, as found in the LICENSE file. [Terms of use](https://opensource.facebook.com/legal/terms). [Privacy](https://opensource.facebook.com/legal/privacy)

## Citations

If you find this code useful in your research then please cite:

```
@article{ridgesfm2020,
  title={RidgeSfM: Structure from Motion via Robust Pairwise Matching Under Depth Uncertainty},
  author={Benjamin Graham and David Novotny},
  journal={3DV},
  year={2020}
}
```
