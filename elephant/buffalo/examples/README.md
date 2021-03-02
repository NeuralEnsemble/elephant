# Elephant Provenance Tracking module
## Examples of usage

This folder contains several scripts that show how the provenance tracking
functionality should be used when implementing analysis scripts using Elephant.

Some examples use generated data, but others work with publicly available
electrophysiology datasets.

## Downloading the datasets

The Reach2Grasp experimental dataset (Brochier et al. Scientific Data 5: 
180055, 2018) is used in some examples. The repository must be downloaded to
your local computer. To download, please follow the instructions on:

https://doi.gin.g-node.org/10.12751/g-node.f83565/

At this website, you can either download a large archive (.zip) or access
the GIN repository where the data is hosted. In the latter case, instructions 
to download the data and code using the `gin` client are provided.

## Running the examples

To run the examples that use the experimental dataset, you must add the 
location of the `reachgraspio.py` file in your local computer to the
`PYTHONPATH` environmental variable. This is located in the folder where you
downloaded the Reach2Grasp dataset, in the subfolder `code`.

Example:
```
export PYTHONPATH="$PYTHONPATH:[absolute_path_to_local_folder]/code/reachgraspio"
```

## Description of the examples in the folder

### run_basic.py
Shows basic functionality of the decorator. The script takes one of the
Reach2Grasp datasets as argument.
Usage is 
```
run_basic.py [path_to_dataset]
```

### run_isi_histogram.py
More advanced example showing how to compute some histograms, plot and save 
the result to a file, while tracking and saving provenance information.
The script takes one of the Reach2Grasp datasets as argument.
Usage is 
```
run_isi_histogram.py [path_to_dataset]
```

### run_time_histogram_old.py
Example showing the functionality of the data analysis objects.
The code shows compares the use of Elephant when using the old and the new
implementation (based on the `TimeHistogramObject`), to showcase compatibility
with existing code.

### run_time_histogram.new.py
Example using the new implementation that uses the data analysis objects,
showing how the embedded provenance information and standardization helps with
the plotting function.

### run_psth.py
Example showing the use of `PSTHObject` and reuse of a data analysis object.

### run_psd.py
Example of the usage of the object that represents power spectrum densities
(`PSDObject`) together with the integration with the provenance tracking 
decorator.
