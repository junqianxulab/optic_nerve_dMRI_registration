# Optic Nerve dMRI Registration<sup>[1](#kim)</sup>
## Non-linear movement/distortion correction in human optic nerve diffusion imaging.

---

## Installation
#### with docker
* run `docker build -t on_reg .`

#### without docker
* install Python 2.7 and its libraries
  * numpy
  * scipy
  * nibabel >= 2.0
  * Priority dictionary (priodict.py, http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/117228)
  * ANTs 2.2.0

---

## Usage:

### [Preparation](data_preparation.md)

### Registration

* example data file names for following commands:
    * fMRI: `dwi.nii.gz`

1. Create an initial ROI file (`dwi_on_init.nii.gz`) containing the followings:

    * posterior end voxel of right optic nerve: one voxel of value 1
    * posterior end voxel of left optic nerve: one voxel of value 2
    * anterior end region (right before eyeball) of optic nerve: 4-5 voxels of value 4 at each of left and right optic nerve on a coronal slice
    * Note: we recommend users set these points on both the mean b0 image and the mean high-b-value image.

2. Create an exclusion ROI file (`dwi_exclusion.nii.gz`) containing masks of recti muscle near optic nerve.

3. Run the following command
```
on_reg.py \
    -i dwi_on_init.nii.gz \
    -e dwi_exclusion.nii.gz \
    dwi.nii.gz
```
4. Check the initial centerlines for all volumes in the output file of step 3 (`dwi_on_centerline.nii.gz` and its dilated image `dwi_on_centerline_dilated.nii.gz`). If some centerline is incorrect, modify exclusion ROI of 2 and re-run 3.

5. Run the following command (the same as step 3 but it reads the created initial centerline generated at step 4)
```
on_reg.py \
    -i dwi_on_init.nii.gz \
    -e dwi_exclusion.nii.gz \
    dwi.nii.gz
```

6. Check the resulting optic nerve segmentation (`dwi_on_model.nii.gz` and its projection image `dwi_on_model_1st_bin.nii.gz`). e.g., `fslview dwi dwi_on_model -l Red -b 0.8,1.2 -t 0.3 dwi_on_model_1st_bin -l Red -t 0.3`. If necessary, modify the exclusion ROI file and re-run step 5.

7. Check the resistered image (`dwi_nonlin_reg.nii.gz`)

### Optic nerve center voxel generation

* Create an optic nerve center ROI file (`on_center.nii.gz`) from the model file (`dwi_on_model`) at step 5.

```
on_create_center_from_model.py \
    -o on_center.nii.gz \
    -f on_center_volumes.nii.gz \
    -r dwi.nii.gz \
    dwi_on_model
```

## Reference:
<a name="Kim">[1]</a> Kim et al, Incorporating non-linear alignment and multi-compartmental modeling for improved human optic nerve diffusion imaging. Neuroimage, 2019, 196:102-113. https://www.ncbi.nlm.nih.gov/pubmed/30930313

