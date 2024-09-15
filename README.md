# ZS-Denoiser-HREM

This repository is Pytorch implementation of our manuscript "Zero-shot Image Denoising for High-Resolution Electron Microscopy" [[ArXiv](https://arxiv.org/abs/2406.14264)][[IEEE Xplore](https://ieeexplore.ieee.org/document/10675590)].

## Pipeline of ZS-Denoiser 


  ![Pipeline_ZS_Denoiser-HREM](Fig/Pipeline.png)
  Fig. 1: The pipeline of ZS-Denoiser HREM.

---

## Simulated HREM Denoising Example
 The simulated TEM dataset released by Mohan *et al*. [[Github](https://github.com/sreyas-mohan/electron-microscopy-denoising)] which consists of approximate 18000 simulated images.
 ![Simulated_Denoising](Fig/res_fig1.png)
 Fig. 2: Comparison of denoising results of simulated Pt/CeO2 catalyst corrputed with Poisson-Gaussain noise.

## Real HREM Denoising Example
 ![Real_STEM_Denoising](Fig/res_fig2.png)
 Fig. 3 Comparison of denoising results of real STEM data on zeolites.
 
---

## 1. Running Environment
To run this project, you will need the following packages:
  
  - Pytorch
  - Scikit-image
  - Tiffile, tqdm, numpy and other packages.
  
## 2. File Tree

```text
ZS-Denoiser-HREM
│  dataset.py 
│  netarch.py             
│  README.md
│  train.py               # train zero-shot denoising network
│  utils.py
│          
├─config
│      simulated_PG.json  # configuration file
│      
└─demo_data
     └─PtCeO2_simulated   # simulated data for numerical experiments
            1.tif
            2.tif
            3.tif
            4.tif
            5.tif

```

## 3. Train the ZS-Denoiser on simulated HREM image

To train the denoising model for simulated HREM image corrupted with Poission-Gaussain noise ($a = 0.05, b = 0.02$), you can run the following command in your terminal:

```shell
python train.py -image_path demo_data/PtCeO2_simulated/1.tif -a 0.05 -b 0.02
```

## 4. License

This code is available for non-commercial research and education purposes only. It is not allowed to be reproduced, exchanged, sold, or used for profit.

## 5. Citation
If you find our work useful in your research, please site:

```
@ARTICLE{10675590,
  author={Tian, Xuanyu and Dong, Zhuoya and Lin, Xiyue and Gao, Yue and Wei, Hongjiang and Ma, Yanhang and Yu, Jingyi and Zhang, Yuyao},
  journal={IEEE Transactions on Computational Imaging}, 
  title={Zero-Shot Image Denoising for High-Resolution Electron Microscopy}, 
  year={2024},
  volume={},
  number={},
  pages={1-14},
  keywords={Noise measurement;Noise reduction;Training;Image denoising;Signal to noise ratio;Electrons;Imaging;Zero-shot;Electron Microscopy;Denoising;Self-supervised},
  doi={10.1109/TCI.2024.3458411}}

```
