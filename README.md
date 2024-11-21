# DSLRDNet

This is the official implementation for the paper "Addressing Multiple Salient Object Detection via Dual-Space Long-Range Dependencies", accepted by the Journal of Computer Vision and Image Understanding (CVIU 2023).

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
1. Download the [pretrained models](https://drive.google.com/file/d/1AdcqpcwIzfLTu4qkUmpPhYKNFqDMv3Sq/view?usp=sharing) (Google Drive).
2. Change the data path in dataset.py
3. Change the test settings in run.py.
4. Generate saliency maps with `python3 run.py --mode test --sal_mode m`, where 'm' demonstrates the MSOD dataset.
5. We use the public open source evaluation code. (https://github.com/weijun88/F3Net)

**Datasets and results:**

[MSOD dataset](https://drive.google.com/file/d/1vzaYNhc8nIS_U4xataDVOqedES6bXlNc/view?usp=drive_link) || [Generated Saliency Maps](https://drive.google.com/file/d/1JHD6ilTtPWvWjMOn4siox9-fyv3UlRSI/view?usp=sharing)  (Goole Drive)

**Citing DSLRDNet:**
```bibtex
@article{deng2023addressing,
  title={Addressing multiple salient object detection via dual-space long-range dependencies},
  author={Deng, Bowen and French, Andrew P and Pound, Michael P},
  journal={Computer Vision and Image Understanding},
  volume={235},
  pages={103776},
  year={2023},
  publisher={Elsevier}
}
