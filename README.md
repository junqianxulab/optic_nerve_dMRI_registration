# Optic Nerve dMRI Registration
## Non-linear movement/distortion correction in human optic nerve diffusion imaging.

Dependency:
#### with docker
* Docker

#### without deocker
* Python 2.7
  * numpy
  * scipy
  * nibabel >= 2.0
  * Priority dictionary (priodict.py, http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/117228)
  * ANTs 2.2.0

## Usage:

### Registration

1. Create an initial ROI file containing the followings:

    * posterior end voxel of right optic nerve: value 1
    * posterior end voxel of left optic nerve: value 2
    * anterior end region (right before eyeball) of optic nerve: value 4

2. Create an exclusion ROI file containing masks of recti muscle near optic nerve.
3. Run the following command
```
on_reg.py \
    -i initial_roi \
    -e exclusion_roi \
    dMRI_filename
```
4. Check created initial centerline files.
5. Run the following command (the same as step 3 but it reads the created initial centerline)
```
on_reg.py \
    -i initial_roi \
    -e exclusion_roi \
    dMRI_filename
```
6. Check the resulting optic nerve segmentation. If necessary, modify the exclusion ROI file and re-run step 5.

### Optic nerve center voxel generation

* Create an optic nerve center ROI file (on_model) from the model file at step 5.

```
on_create_center_from_model.py \
    -o output_filename \
    -f output_filename_for_centerline_for_frames_before_registration \
    -r reference_nifti_(dMRI_filename) \
    model_filename
```

## Reference:
[1] Kim et al, Incorporating non-linear alignment and multi-compartmental modeling for improved human optic nerve diffusion imaging. Neuroimage, 2019, 196:102-113. https://www.ncbi.nlm.nih.gov/pubmed/30930313

