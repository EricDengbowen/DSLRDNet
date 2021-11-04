# DSLRDNet

DSLRDNet: Addressing Multiple Salient Object Detection via Dual-Space Long-Range Dependencies

**Prerequisites:**
1. Pytorch 1.2.0
2. Opencv 2.4.5
3. TensorboardX

**For training:**
1. Download the [DUTS-TR](https://drive.google.com/file/d/1B8AfXfoBWqQ6Zot0NZBwJYVKxWtrpqxF/view?usp=sharing) (Google Drive) training dataset.
2. Download the initial pratrained [VGG/ResNet](https://drive.google.com/drive/folders/1Olx7bugmBCmh4s5AdppHABzMRT3ja9FK?usp=sharing) (Google Drive) model.
3. Change the training data path in dataset.py.
4. Change the training settings in solver.py and run.py
5. Start to train with `python3 run.py --mode train`

**For testing:**
1. Download the [pretrained models](http://cvl.cs.nott.ac.uk/resources/dslrd.pretrained.zip) (UoN server).
2. Change the data path in dataset.py
3. Change the test settings in run.py.
4. Generate saliency maps with `python3 run.py --mode test --sal_mode m`, where 'm' demonstrates the MSOD dataset.
5. We use the public open source evaluation code. (https://github.com/weijun88/F3Net)

**Datasets and results:**

[MSOD dataset](http://cvl.cs.nott.ac.uk/resources/msod.dataset.zip) || [Generated Saliency Maps](http://cvl.cs.nott.ac.uk/resources/saliency.maps.zip)
