# image segmentation

For the fourth SuperComputing project, the concepts of GPGPU seen in class were applied to a real image segmentation problem. Image segmentation is its division into connected regions (without interruptions) containing objects of interest or the image's background. Iterative image segmentation is based on adding markers to objects of interest to differentiate them from the background (which contains all other objects and mage's background). 

## Compile and run

To compile both the sequential program and the GPU program, run the following command:

```
make
```

This command will generate two executables, ./seq and ./gpu. To run them run as follows:

```
<exe> <input_image.pgm> <out_image.pgm> < <input_file.txt>
```

There are examples of input images in the input_images folder and input files in the input_files folder

## Input file

Input files must have the following format:

```
n_forward_seeds n_background_seeds
x_0_forward_seed y_0_forward_seed
x_1_forward_seed y_1_forward_seed
.
.
.
x_n-1_forward_seed y_n-1_forward_seed
x_0_background_seed y_0_background_seed
x_1_background_seed y_1_background_seed
.
.
.
x_n-1_background_seed y_n-1_background_seed
```

## Time analysis
To get the vectors for time analysis explained in the ipython notebook from the "Relatorio" folder, run the following command:

```
python run.py
```

the vectors will be written in the analysis.txt file
