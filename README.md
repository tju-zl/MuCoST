# MuCoST

## Overview
![](https://github.com/tju-zl/MuCoST/blob/master/framework.eps)

Spatially resolved transcriptomics data are being used in a revolutionary way to decipher the spatial pattern of gene expression and the spatial architecture of cell types. Much work has been done to exploit the genomic spatial architectures of cells. Such work is based on the common assumption that gene expression profiles of spatially close spots are more similar than those of more distant spots. However, such methods might not capture the nonlocal spatial dependency of tissue architectures. Therefore, we propose MuCoST, a Multi-view graph Contrastive learning framework for deciphering complex Spatially resolved Transcriptomic architectures with a two-scale structural dependency. To achieve this, we employ spot dependency augmentation by fusing gene expression similarity and proximity of spatial locations, thereby enabling MuCoST to model both gene expression dependency and spatial dependency, while, at the same time, preserving the intrinsic representation of individual cells/spots. We benchmark MuCoST on human brain and mouse brain tissues, and we compare it with other state-of-the-art spatial domain identification methods. We demonstrate that MuCoST achieves the highest accuracy on spatial domain identification and that it can characterize subtle spatial architectures consistent with biological annotations.

## Getting started
See [Documentation and Tutorials](https://github.com/tju-zl/MuCoST/blob/master/MuCoST_DLPFC%20Tutorial.ipynb)

## SRT data collection for evaluating MuCoST
All datasets are open access. The DLPFC datasets are available online at http://research.libd.org/spatialLIBD/. Mouse brain posterior and coronal mouse brain datasets are available online at https://www.10xgenomics.com/. Mouse olfactory bulb of Stereo-seq dataset is available online at https://github.com/JinmiaoChenLab/SEDR_analyses. We provide a collection of the above data at https://zenodo.org/record/8303947.

## Expansibility with histology
In order to use the structural information of histological images, we also provide a histological mode, and the preprocessing of histological images is basis on the [tutorial of SpaGCN](https://github.com/jianhuupenn/SpaGCN/blob/master/tutorial/tutorial.ipynb). Set `mode_his` to `his` to open this mode. We do not enable this mode in the evaluation.

## Software depdendencies
- Scanpy 1.9.3
- PyG 2.3.0
- Pytorch 2.0.1
- scikit-learn 1.2.2
- numpy 1.23.4
- tqdm 4.65.0
- numba 0.56.4
- ot 0.9.0
- argparse 1.1
- r-base 3.6.1
