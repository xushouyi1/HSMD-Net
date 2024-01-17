# HSMD-Net
 Recently, U-shaped neural networks have gained widespread application in remote sensing image dehazing and achieved promising performance. However, most of the existing U-shaped dehazing networks neglect the global and local information interaction across layers during the encoding phase, which leads to incomplete utilization of the extracted features for image restoration. Moreover, in the process of image reconstruction, utilizing only the information from the terminal layers of the decoding phase for haze-free image restoration leads to a dilution of semantic information, resulting in color and texture deviations in the dehazed image. To address these issues, We propose a Hierarchical Slice Interaction and Multi-layer Cooperative Decoding Networks for Remote Sensing Image dehazing(HSMD-Net). Specifically, a Hierarchical Slice Information Interaction Module(HSIIM) is proposed to introduce Intra-layer feature autocorrelation and Inter-layer feature cross-correlation to facilitate global and local information interaction across layers, thereby enhancing the encoding features representation capability and improving the network’s dehazing performance. Furthermore, a Multi-layer Cooperative Decoding Reconstruction module(MCDRM) is proposed to fully utilize feature information in each decoding layer, mitigate semantic information dilution, and improve the network’s capability to restore image colors and textures. The experimental results demonstrate that our HSMD-Net outperforms several state-of-the-art methods in dehazing on two publicly available datasets. 
<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]


<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/xushouyi1/HSMD-Net/">
    <img src="images/Framework.png" alt="Logo" width="1000" height="450">
  </a>
  <h3 align="left">Hierarchical Slice Interaction and Multi-layer Cooperative Decoding
Networks for Remote Sensing Image dehazing</h3>
  <p align="center">
  <a href="https://github.com/xushouyi1/HSMD-Net/">
    <img src="images/HIISM.png" alt="Logo" width="1000" height="500">
  </a>
  </p>
  <h3 align="center">Hierarchical Slice Information Interaction module</h3>
  <p align="center">
  <a href="https://github.com/xushouyi1/HSMD-Net/">
    <img src="images/MDCRM.png" alt="Logo" width="1200" height="250">
  </a>
  </p>
  <h3 align="center">Multi-layer Cooperative Decoding Reconstruction module</h3>
  <p align="center">
    <br />
    <a href="https://github.com/xushouyi1/HSMD-Net/"><strong>Exploring the documentation for HSMR-Net »</strong></a>
    <br />
    <br />
    <a href="https://github.com/xushouyi1/HSMD-Net/">Check Demo</a>
    ·
    <a href="https://github.com/xushouyi1/HSMD-Net/">Report Bug</a>
    ·
    <a href="https://github.com/xushouyi1/HSMD-Net/">Pull Request</a>
  </p>

</p>

## Contents
- [Dependencies](#dependences)
- [Filetree](#filetree)
- [Pretrained Model](#pretrained-weights-and-dataset)
- [Train](#train)
- [Test](#test)
- [Clone the repo](#clone-the-repo)
- [Qualitative Results](#qualitative-results)
  - [Results on Statehaze1k-Thin remote sensing Dehazing Challenge testing images:](#results-on-statehaze1k-thin-remote-sensing-dehazing-challenge-testing-images)
  - [Results on Statehaze1k-Moderate remote sensing Dehazing Challenge testing images:](#results-on-statehaze1k-moderate-remote-sensing-dehazing-challenge-testing-images)
  - [Results on Statehaze1k-Thick remote sensing Dehazing Challenge testing images:](#results-on-statehaze1k-thick-remote-sensing-dehazing-challenge-testing-images)
  - [Results on HRSD-DHID remote sensing Dehazing Challenge testing images:](#results-on-hrsd-dhid-remote-sensing-dehazing-challenge-testing-images)
  - [Results on HRSD-LHID remote sensing Dehazing Challenge testing images:](#results-on-hrsd-lhid-remote-sensing-dehazing-challenge-testing-images)
- [Thanks](#thanks)

### Dependences

1. Pytorch 1.8.0
2. Python 3.7.1
3. CUDA 11.7
4. Ubuntu 18.04
### Filetree

```
├── README.md
├── /HSMD-Net/
|  ├── train.py
|  ├── test.py
|  ├── Model.py
|  ├── Model_util.py
|  ├── perceptual.py
|  ├── train_dataset.py
|  ├── test_dataset.py
|  ├── utils_test.py
|  ├── make.py
│  ├── /pytorch_msssim/
│  │  ├── __init__.py
│  ├── /datasets_train/
│  │  ├── /hazy/
│  │  ├── /clean/
│  ├── /datasets_test/
│  │  ├── /hazy/
│  │  ├── /clean/
│  ├── /output_result/
|  |  ├── ../pkl
└── /images/
```
### Pretrained Weights and Dataset

Download our model weights on Baidu cloud disk: https://pan.baidu.com/s/15gOQ9qADAtDfgY9dkIvOxA?pwd=39ni 

Download our test datasets on google cloud disk: https://drive.google.com/drive/folders/1SRPc3RPdXZQofQn4lwhQDNHTQeNRJWfl?usp=sharing

### Train

```shell
python train.py -train_batch_size 4 --gpus 0 --type 0
```

### Test

 ```shell
python test.py --gpus 0 --type 0
 ```

### Clone the repo

```sh
git clone https://github.com/xushouyi1/HSMD-Net.git
```

### Qualitative Results

#### Results on Statehaze1k-Thin remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/Thin_Fog.png" style="display: inline-block;" />
</div>

#### Results on Statehaze1k-Moderate remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/Moderate_Fog.png" style="display: inline-block;" />
</div>

#### Results on Statehaze1k-Thick remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/Thick_Fog.png" style="display: inline-block;" />
</div>

#### Results on HRSD-LHID remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/LHID.png" style="display: inline-block;" />
</div>

#### Results on HRSD-DHID remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/DHID.png" style="display: inline-block;" />
</div>




### Thanks


- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)


<!-- links -->
[your-project-path]:thislzm/PSMB-Net
[contributors-shield]: https://img.shields.io/github/contributors/xushouyi1/HSMD-Net.svg?style=flat-square
[contributors-url]: https://github.com/xushouyi1/HSMD-Net/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/xushouyi1/HSMD-Net.svg?style=flat-square
[forks-url]: https://github.com/thislzm/xushouyi1/HSMD-Net/members
[stars-shield]: https://img.shields.io/github/stars/xushouyi1/HSMD-Net.svg?style=flat-square
[stars-url]: https://github.com/xushouyi1/HSMD-Net/stargazers
[issues-shield]: https://img.shields.io/github/issues/xushouyi1/HSMD-Net.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/xushouyi1/HSMD-Net.svg
[license-shield]: https://img.shields.io/github/license/xushouyi1/HSMD-Net?style=flat-square
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian
