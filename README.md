

# DENCLUE from scratch

This is a project that implements DENCLUE (Density-Based Clustering Algorithm with Noise), a density-based clustering algorithm, from scratch using Python and scikit-learn. This project also includes the generation of the dataset used for testing.

## Dataset generation

To generate the dataset follow the following steps.

1. Install scikit-learn on your system.

```
pip install scikit-learn
```

2. Generate 2D data similar to the one shown in the third row of the image at the link https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html. You can use a scikit function for data generation. Data should have at least 200 datapoints and should be 2D.

3. Save the generated dataset in a csv file.

## DENCLUE Algorithm

1. Read the data from the csv file.

2. Code your own DENCLUE function which takes the data, the threshold T and the parameter h (for a Gaussian kernel function) and returns the clusters output.

3. Plot the clusters using different colors.

## About DENCLUE

A description of DENCLUE can be found at https://cs.nju.edu.cn/zlj/Course/DM_15_Lecture/Lecture_6.pdf.
