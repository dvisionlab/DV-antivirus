# DV-antivirus

Lung segmentation and statistical analisys

## Goal

The main purpose of the code in this repo is to calculate the percentage of lung volume where the perfusion is lower than a user-defined threshold.
In order to reach this goal a number of scripts have been implemented. They permit to:

- divide DICOM study folders into DICOM series folders
- perform lungs segmentation (thanks to [lungmask](https://github.com/JoHof/lungmask))
- isolate the lungs pixels values and report them into a .csv file, divided in left and right lung
- generate histograms from .csv file, computing the ratio between pixels lower / higher than the threshold

Moreover, a set of side scripts have been implemented to further analysis and/or speed up the workflow:

- images registration
- folders unzip

In particular, at the moment the best results are obtained using images with contrast medium. The future aim is to find a way to obtain the same results without the use of contrast medium.

## Installation

### Linux

> pipenv install

That's all!

## Windows

- Install [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)
  [non rimpiazzare il python di default,
  non aggiungere nulla alle variabili d ambiente]
- Install pytorch with `conda install pytorch torchvision cpuonly -c pytorch`
- Install needed packages (see Pipfile) with `conda install` or `pip install`

eg:

> conda install -c conda-forge pydicom  
> conda install -c simpleitk simpleitk  
> conda install -c anaconda scikit-image

## Usage

### Folders subdivision

Run this to subdivide series in the main folder:

> python utils.py --organize path/to/dicom/study/folder

### Lung segmentation & values extraction

Run this to segment lungs and extract values to .csv file:

> python run.py --dicomdir path/to/dicom/series/folder  
> Options:  
> --thresholds [lower_bound, threshold, upper_bound] : use this to set custom thresholds (must be same as next step)  
> --nrrddir path/to/nrrd/file : use instead of --dicomdir to read a .nrrd file as input  
> --outfolder path/to/output/folder : specify output folder (must exist), default ./output/  
> --force_cpu : set this to force prediction with cpu (slower), not needed if no gpu is available  
> --use_mask path/to/lung/mask : use this to pass a previously generated mask and skip segmentation

The output consist in a nrrd file (lung_mask_palette.nrrd) and output.csv file.
Note: the input dicom serie is converted in .nrrd file in the ./temp/ directory, to be used as input to
segmentation neural net.

### Histograms calculation

Run this to generate histograms and compute low perfusion ratio:

> python utils.py --examine path/to/csv  
> Options:  
> --thresholds [lower_bound, threshold, upper_bound] : use this to set custom thresholds (must be same as previous step)

Histogram is shown and saved in the same directory of the input file, as histogram.png and histogram.csv.

## Authors

You can contact us at:

mattia.ronzoni@dvisionlab.com

www.dvisionlab.com
