# TFZ: Topology-Preserving Augmentation of Lossy Compressors for 2D Tensor Fields

Related Publication:

- Nathaniel Gorski, Xin Liang, Hanqi Guo, and Bei Wang. TFZ: Topology-Preserving Compression of 2D Symmetric and
Asymmetric Second-Order Tensor Fields. IEEE TVCG 2026.

[arxiv](https://www.arxiv.org/abs/2508.09235)

## Overview

In this repo we provide the main compressor used in the related publication, including the setup needed to replicate the experiments. We also include the scripts used to generate the hyperLIC and partition visualizations.

### Data Format

As in the publication, we represent each tensor field as a flat triangular mesh. We use a rectilinear mesh. Our mesh takes the form of a grid of squares. Each square is further subdivided into two triangles as follows: if one of the squares is the unit square, the two triangular cells that it contains will have vertices {(0,0), (0,1), (1,0)} and {(1,0), (0,1), (1,1)}. We store the value of the tensor at each vertex of the grid, and use linear interpolation for all interior points.

Our support tensor field data stores in RAW binary format. A tensor field is stored as a directory containing four files: ```A.raw```, ```B.raw```, ```C.raw```, and ```D.raw```. If 

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/sf-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="figures/sf.png">
  <img alt="Example" src="sf.png", width="200">
</picture>
</p>

is the tensor field, then our files respectively store the values of the scalar fields $A$, $B$, $C$, and $D$ at each vertex point. Internally, the name of the tensor field is referenced by the name of the directory.

$A$, $B$, $C$, and $D$ should be stored as double precision floating point values stored in little endian. The scalar fields should be stored in column-major (Fortran) order.

In the publication we worked with collections of 2D scalar fields. To work with a collection, simply concatenate the values of ```A.raw``` ```B.raw``` ```C.raw``` and ```D.raw``` for each tensor field in the collection. To work with collections in this way, each tensor field must have the same dimensions. We will refer to each tensor field in a collection as a "slice."

## Installation

Our implementation is written in pure Julia. **Recent updates to Julia's memory management system hurt the performance of our implementation. For best results, use Julia 10.4**.

### Dependencies:

```
DataStructures.jl
StaticArrays.jl
WriteVTK.jl
```

```WriteVTK.jl``` is only required for the visualization scripts and is not required for TFZ itself.