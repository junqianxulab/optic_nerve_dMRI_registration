#!/usr/bin/env python

import sys
import os
import subprocess
import nibabel as nib
import numpy as np

def get_frames_b0_dwi(filename, threshold_b0=90, threshold_min=500, threshold_max=900, return_verbose=False, verbose=True):
    fin = open(filename)
    frames = [int(value) for value in fin.readline().strip().split()]
    frames_b0 = []
    frames_dwi = []
    rtn_str = '# '
    for i in range(len(frames)):
        if frames[i] < threshold_b0:
            frames_b0.append(i)
            rtn_str += str('[%s]:%s ' % (i,frames[i]))
        if threshold_min <= frames[i] <= threshold_max:
            frames_dwi.append(i)
            rtn_str += str('(%s):%s ' % (i,frames[i]))
        else:
            rtn_str += str('%s:%s ' % (i,frames[i]))
        if (i+1)%5 == 0:
            rtn_str += '\n# '
    rtn_str += '\n'

    fin.close()
    if verbose:
        print rtn_str
    if return_verbose:
        return frames_b0, frames_dwi, rtn_str
    return frames_b0, frames_dwi

def filename_wo_ext(filename):
    if filename[-7:] == '.nii.gz':
        return filename[:-7]
    if filename[-4:] == '.nii':
        return filename[:-4]
    return filename

if len(sys.argv) < 3:
    sys.stderr.write('%s dwi_name bval [output_basename [threshold_b0=90 threshold_min=500 threshold_max=900]]\n' % sys.argv[0])
    sys.exit(-1)

filename = sys.argv[1]
filename_bval = sys.argv[2]
if len(sys.argv) > 3:
    filename_out = filename_wo_ext(sys.argv[3])
else:
    filename_out = filename_wo_ext(filename)
if len(sys.argv) > 4:
    threshold_b0 = int(sys.argv[4])
    threshold_min = int(sys.argv[5])
    threshold_max = int(sys.argv[6])
else:
    threshold_b0 = 90
    threshold_min = 500
    threshold_max = 900


if not os.path.isfile(filename):
    print '%s not found.' % filename
    sys.exit(-1)
if not os.path.isfile(filename_bval):
    print '%s not found.' % fn_mask
    sys.exit(-1)

# extract
img = nib.load(filename)
aff = img.get_affine()
hdr = img.get_header()
dat = img.get_data()
nt = img.shape[3]

frames_b0, frames_dwi, rtn = get_frames_b0_dwi(filename_bval, threshold_b0=threshold_b0, threshold_min=threshold_min, threshold_max=threshold_max, return_verbose=True, verbose=True)
print ' frames_b0: ', frames_b0
print ' frames_dwi: ', frames_dwi

dat_out = np.zeros( img.shape[:3], dtype=np.float)
dat_out_b0 = np.zeros( img.shape[:3], dtype=np.float)
dat_out_c = np.zeros( img.shape[:3]+(len(frames_dwi),), dtype=np.float)
i_b0 = 0
i_dwi = 0
for t in range(nt):
    if t in frames_b0 or str(t) in frames_b0:
        dat_out_b0[:,:,:] += dat[:,:,:,t]
        i_b0+=1

    if t in frames_dwi or str(t) in frames_dwi:
        dat_out[:,:,:] += dat[:,:,:,t]
        dat_out_c[:,:,:,i_dwi] = dat[:,:,:,t]
        i_dwi+=1

dat_out_b0 /= i_b0
hdr.set_data_dtype(dat_out_b0.dtype)
img_out = nib.Nifti1Image(dat_out_b0, aff, hdr)
nib.save(img_out, filename_out+'_b0.nii.gz')

dat_out /= i_dwi
hdr.set_data_dtype(dat_out.dtype)
img_out = nib.Nifti1Image(dat_out, aff, hdr)
nib.save(img_out, filename_out+'_dwi.nii.gz')

if False:
    hdr.set_data_dtype(dat_out_c.dtype)
    img_out = nib.Nifti1Image(dat_out_c, aff, hdr)
    nib.save(img_out, filename_out+'_dwi_frames.nii.gz')

