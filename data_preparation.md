## Data Preparation for Optic Nerve dMRI Registration

The optic nerve dMRI runs on a single 4-D dMRI file. If there are multiple session images, they need to be merged first.

### 1) using FSL TOPUP/EDDY

* FSL TOPUP/EDDY corrects susceptibility- and eddy current-induced distortions as well as head motions. Its output is a single, merged dMRI file.
* Since TOPUP/EDDY is for brain dMRI, it is not as robust on optic nerve dMRI as brain dMRI. Hence, QA is critical and if the result is not satisfactory, you better use an alternative method, 2).

### 2) rigid-body regisration among sessions

* This is to correct bulk head motion among sessions based on brain structures other than optic nerve.
1. Get a representative image from each session. This can be done by averaing all b0 images of each session.
2. Create a reference image for the rigid-body registration. The can be an average image of all representative images or a representative image.
3. Register each representative image to the reference image using ANTs, flirt, or other tools.
4. Apply the transform at 3 of each session to all volumes in the session.
5. Merge the sessions using fslmerge or other similar tools.

* The above steps can be done using `on_rigid_reg.py` script.
1. Create `dwi_filelist.txt` file containing dMRI filenames. e.g.,
```
sub01_dwi_1_PA.nii.gz
sub01_dwi_2_PA.nii.gz
```
2. Run the following command in a subdirectory `reg/rigid`
```
on_rigid_reg.py ../../dwi_filelist.txt
```
3. There will be command outputs to create a mask file. set proper threshold value (`thr=200` or `thr=2000`), the run the commands.
4. Check the created mask file using a viewer (e.g., `fslview` or `fsleyes`) and modify it if necessary.
5. Run the same command as 2 again
```
on_rigid_reg.py ../../dwi_filelist.txt
```
6. Check the registration of the output files (e.g., `sub01_dwi_1_PA_rigid.nii.gz` and `sub01_dwi_2_PA_rigid.nii.gz`)

7. Merge the registered dwi files.
```
fslmerge -t dwi_rigid sub01_dwi_1_PA_rigid.nii.gz sub01_dwi_2_rigid.nii.gz
```

8. Merge b-value and b-vector files.
```
paste sub01_dwi_1_PA.bval sub01_dwi_2_PA.bval >dwi_rigid.bval
paste sub01_dwi_1_PA.bvec sub01_dwi_2_PA.bvec >dwi_rigid.bvec
```

