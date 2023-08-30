# MuCoST

## Overview
![](https://github.com/tju-zl/MuCoST/blob/master/framework.png)

Spatially resolved transcriptomics data are being used in a revolutionary way to decipher the spatial pattern of gene expression and the spatial architecture of cell types. Much work has been done to exploit the genomic spatial architectures of cells. Such work is based on the common assumption that gene expression profiles of spatially close spots are more similar than those of more distant spots. However, such methods might not capture the nonlocal spatial dependency of tissue architectures. Therefore, we propose MuCoST, a Multi-view graph Contrastive learning framework for deciphering complex Spatially resolved Transcriptomic architectures with a two-scale structural dependency. To achieve this, we employ spot dependency augmentation by fusing gene expression similarity and proximity of spatial locations, thereby enabling MuCoST to model both gene expression dependency and spatial dependency, while, at the same time, preserving the intrinsic representation of individual cells/spots. We benchmark MuCoST on human brain and mouse brain tissues, and we compare it with other state-of-the-art spatial domain identification methods. We demonstrate that MuCoST achieves the highest accuracy on spatial domain identification and that it can characterize subtle spatial architectures consistent with biological annotations.



## Getting started
See [Documentation and Tutorials] (https://github.com/tju-zl/MuCoST/blob/master/MuCoST_DLPFC%20Tutorial.ipynb)

