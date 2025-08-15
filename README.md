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
  <img alt="Example" src="sf.png", width="150">
</picture>
</p>

is the tensor field, then our files respectively store the values of the scalar fields $A$, $B$, $C$, and $D$ at each vertex point. Internally, the name of the tensor field is referenced by the name of the directory.

$A$, $B$, $C$, and $D$ should be stored as double precision floating point values stored in little endian. The scalar fields should be stored in column-major (Fortran) order.

In the publication we worked with collections of 2D scalar fields. To work with a collection, simply concatenate the values of ```A.raw``` ```B.raw``` ```C.raw``` and ```D.raw``` for each tensor field in the collection. To work with collections in this way, each tensor field must have the same dimensions. We will refer to each tensor field in a collection as a "slice."

## Installation

Our implementation is written in pure Julia. **Recent updates to Julia's memory management system hurt the performance of our implementation. For best results, use Julia 10.4**. Our implementation has only been tested on Linux. We believe that it would also work on macOS, but it would not work on Windows.

### Dependencies:

```
DataStructures.jl
StaticArrays.jl
WriteVTK.jl
```

```WriteVTK.jl``` is only required for the visualization scripts and is not required for TFZ itself.

## Usage

### TFZ:

Run ```julia tfz.jl <arguments>```. All options can be viewed by running ```julia tfz.jl -h```. We provide additional clarifications here.

- TFZ makes a lot of system calls, so by default ```stdout``` and ```stderr``` are suppressed. To view any printouts (such as in the event of a crash), use ```-verbose```.

#### Specifying Files

- When specifying an input with the ```-i``` flag, list the name of the directory that contains ```A.raw```, ```B.raw```, ```C.raw```, and ```D.raw```.
- Suppose that you specify compressed name `Z` using the `-z` flag and decompressed name `O` using the `-o` flag:
    - If you are compressing a single slice, TFZ will create a single compressed file called `Z.tar.zst`. The decompressed file will be a directory called `O` containing ```A.raw```, ```B.raw```, ```C.raw```, and ```D.raw```.
    - If you are compressing many slices, TFZ will create a directory called `Z` that will contain many files called ```slice_01.tar.zst```, ```slice_ot.tar.zst``` etc. corresponding to each slice that was compressed. Similarly, an output directory called `O` will be created. It will contain subdirectories called ```slice_01```, ```slice_02``` etc. each containg the ```A.raw```, ```B.raw```, ```C.raw``` and ```D.raw``` corresponding to each decompressed slice.
    - If you are only decompressing `-z` should either specify a `.tar.zst` file corresponding to a single compressed slice, or a directory containing files named ```slice_01.tar.zst```, ```slice_02.tar.zst``` etc.
- In order to compress multiple slices simultaneously, use ```-n_slices``` to specify how many slices are being compressed. To compress one slice out of a collection, specify which one using the ```-slice``` flag.
    - Because Julia is 1-indexed, the first slice has index 1.
    - To decompress a single file out of a collection, specify the name of the ```.tar.zst``` file rather than using ```-slice```.

#### Experiment Flag

- Use ```-experiment``` to collect data using TFZ.
- Use the ```-csv``` flag to save the results of an experiment to a CSV file. If the CSV file already exists, a new row will be appended to the end containing the information from the current experiment.
- By default all of the evaluation metrics reported in the paper will be collected. A limited subset of the evaluation metrics will be printed to the terminal if the ```-experiment``` flag is not set.
    - If you only need to collect times, using ```-skipStatistics``` will skip computing most statistics, leading to faster experiment times.
- When using ```-experiment```, you do not need to specify anything using ```-z``` or ```-o```. If either flag is omitted, a temporary file will be created for the compressed or decompressed file, respectively, and deleted at the end of the experiment.

## Visualization scripts

The visualization scripts can be found in the ```vis_tools``` folder. These scripts can only be used to visualize one slice at a time. However, one can set the input file equal to a collection of slices and specify a single slice to visualize. Because Julia is 1-indexed, the first slice is considered to have index 1.

### hyperLIC

- To run the hyperLIC script, run ```julia hyperlic.jl <arguments>```. For usage details, run ```julia hyperlic.jl -h```.
- We use a home-built implementation of [fast-LIC](http://www.zhanpingliu.org/research/flowvis/lic/FastLIC/FastLIC.htm) adapted to vector fields.
- In order to spread out the streamlines that we generate, we divide the domain into an $n \times n$ grid, and spread out streamlines across the different grid cells. The number of grid cells in each direction can be specified using ```-block_size```. By default this is set to 20.
- Based on how fastLIC is implemented, the first pixels to be processed take the longest, and pixels take less time to process over the course of the algorithm.
- Two files will be outputted, both of which are to be visualized in ParaView. One is a ```.vti``` file that contains both the hyperLIC values, as well as the Frobenius norm of each tensor. Another is a ```.vtu``` file that contains the locations of the degenerate points.

### Partition Visualization

- To run the visualization script, run ```julia partitions.jl <arguments>```. For usage details, run ```julia partitions.jl -h```
- Two files will be outputted, both of which are to be visualized in ParaView. One is a ```.vti``` file that contains the following arrays:
    - ```categorical val``` : Each mesh vertex is assigned an integer from 1-5 based on its classification in the eigenvalue manifold.
    - ```categorical vec``` : Each mesh vertex is assigned an integer from 1-5 based on its classification in the eigenvector manifold.
    - ```color val``` : Each mesh vertex is assigned three integers corresponding depending on its classification in the eigenvalue manifold. The integers correspond to the RGB color used to visualize that classification in the publication.
    - ```color vec``` : Similar to ```color val``` but for the eigenvector partition.
    - ```frobenius``` : Stores the Frobenius norm of each tensor.
- ```color val``` and ```color vec``` can only be visualized properly by unchecking the ```Map Scalars``` option listed under ```Scalar Coloring```.
- The other file that is outputted is a ```.vtu``` file. It contains the following arrays:
    - ```categorical val``` and ```categorical vec``` : Same as above.
    - ```color criticalType``` : Each point is assigned three integers corresponding to whether it is a trisector or wedge. The integers correspond to the RGB color used to visualize that type of degenerate point in the publication.
    - ```frobenius``` : The Frobenius norm of the tensor field at that critical point.
