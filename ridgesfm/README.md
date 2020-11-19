- Follow instructions in [`data/README.md`](data/README.md`)
- Download [SuperPoint](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.pdf) to ridgesfm/
```
wget https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/weights/superpoint_v1.pth?raw=true -O ridgesfm/weights/superpoint_v1.pth
wget https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/superpoint.py -O ridgesfm/superpoint.py
```
- Run `python ridgesfm.py`