# DrawMicromodels

Notebook to draw micromodels of various shapes and types. 

# Synthetic Multi-Scale Porous Media Generation

This script generates synthetic images of multi-scale porous media that mimic the structural complexity of real porous materials. The generation process consists of three main stages: macropore geometry synthesis, micro-porosity incorporation, and data export.

1. Macropore Geometry Synthesis
The custom class, CircleImageGenerator, is used to synthesize the macroporous phase as overlapping circles randomly distributed across a 2D domain. The circles are rasterized into binary images, where the pores are denoted by 0 (black) and the solid matrix by 255 (white).

2. Micro-Porosity Integration
Micro-porous regions are introduced by morphological transformations on the binary base image. The addMicroPorosity class creates a differential mask by performing morphological opening and closing operations, simulating infiltration of finer pores within the solid phase. The generated mask is then discretised into two micro-porosity levels using intensity binning and iterative morphological adjustments to balance the volumetric fraction of each micro-phase. The final phase-labeled image has four distinct classes: macropores, two micro-porous phases, and solid matrix.

3. Data Export and Labelling
Each generated sample is exported in both .hdf5 and raw binary formats using the export_data class. The .hdf5 files store metadata including coordinate arrays, radius arrays, and the pixel-wise phase distribution. Each sample also includes attributes such as phase volume fractions and geometric parameters used during generation. The full 3D volume (extruded from the 2D image) is saved in raw format for compatibility with simulation tools.
