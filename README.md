**Recent updates to Julia's memory management system hurt the performance of TFZ. For best results, use Julia 10.4**.

# TFZ: Topology-Preserving Augmentation of Lossy Compressors for 2D Tensor Fields

Related Publication:

- Nathaniel Gorski, Xin Liang, Hanqi Guo, and Bei Wang. TFZ: Topology-Preserving Compression of 2D Symmetric and
Asymmetric Second-Order Tensor Fields. IEEE TVCG 2026.

[arxiv](https://www.arxiv.org/abs/2508.09235)

## Overview

In this repo we provide the main compressor used in the related publication, including the setup needed to replicate the experiments. We also include the scripts used to generate the hyperLIC and partition visualizations.

### Data Format

Our support tensor field data stores in RAW binary format. A tensor field is stored as a directory containing four files: ```A.raw```, ```B.raw```, ```C.raw```, and ```D.raw```, which respectively store the $(1,1)$, $(1,2)$, $(2,1)$ and $(2,2)$ entries 