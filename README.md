# MuCoST
## A Multi-view Graph Contrastive Learning Framework for Deciphering Spatially Resolved Transcriptomics Data.

Spatially resolved transcriptomics data is being used in a revolutionary way to decipher the spatial pattern of gene expression and the spatial architecture of cell types. Much work has been done to exploit the genomic spatial architectures of cells. Such work is based on the common assumption that gene expression profiles of spatially adjacent spots are more similar than those of more distant spots. However, related work might not consider the nonlocal spatial co-expression dependency, which can better characterize the tissue architectures. Therefore, we propose MuCoST, a **Mu**lti-view graph **Co**ntrastive learning framework for deciphering complex **S**patially resolved **T**ranscriptomic architectures with dual scale structural dependency. To achieve this, we employ spot dependency augmentation by fusing gene expression correlation and spatial location proximity, thereby enabling MuCoST to model both nonlocal spatial co-expression dependency and spatially adjacent dependency. We benchmark MuCoST on four datasets, and we compare it with other state-of-the-art spatial domain identification methods. We demonstrate that MuCoST achieves the highest accuracy on spatial domain identification from various datasets. In particular, MuCoST accurately deciphers subtle biological textures and elaborates the variation of spatially functional patterns.

## Overview
![](https://github.com/tju-zl/MuCoST/blob/master/framework.png)
**Overview of MuCoST**. MuCoST is a multi-view graph contrastive learning framework for deciphering SRT data. **a. Workflow of MuCoST**. MuCoST uses gene expression profile and spatial location information of SRT data as input data. Spatially adjacent graph, co-expression graph and shuffled graph are constructed in MuCoST. MuCoST adopts a multi-view graph contrastive learning framework to learn latent representations of three views by the InfoNCE loss function. MuCoST extracts the compact latent representation through the reconstruction loss of GCN autoencoder. **b. Downstream Analysis**. MuCoST performs clustering on latent representation of spatial view and realizes downstream analysis tasks, such as spatial cluster visualization, spatial domain identification, PAGA trajectory inference, subtle biological texture recognition and functional analysis of spatial domain.

## Getting started
See [Documentation and Tutorials](https://github.com/tju-zl/MuCoST/blob/master/MuCoST_DLPFC%20Tutorial.ipynb).

## SRT data collection for evaluating MuCoST
All datasets are open access. The DLPFC datasets are available online at http://research.libd.org/spatialLIBD/. The coronal mouse brain of 10X Visium dataset is available online at https://squidpy.readthedocs.io. Mouse olfactory bulb of Stereo-seq dataset is available online at https://github.com/JinmiaoChenLab/SEDR_analyses/. Human breast cancer of 10X Visium dataset is available online at https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/. We provide a collection of part of the above data at https://zenodo.org/record/8303947/.

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
