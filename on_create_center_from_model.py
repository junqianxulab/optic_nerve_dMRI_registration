#!/usr/bin/env python

import argparse
import numpy as np
import nibabel as nib

import on_model

parser = argparse.ArgumentParser()
parser.add_argument('fn_model', help='model_filename')
parser.add_argument('-o', help='output_filename', dest='fn_out', default='on_center.nii.gz')
parser.add_argument('-f', help='output_center_frames_basename', dest='fn_frame_out', default='on_center_frames')
parser.add_argument('-r', help='reference nifti', dest='fn_ref', default=None)

ag = parser.parse_args()

if ag.fn_ref is None:
    import os
    import glob
    import sys
    lst = glob.glob(os.path.join(os.path.dirname(ag.fn_model), '*.nii.gz'))
    if len(lst) == 0:
        sys.stderr.write('no nifti file for reference in %s\n' % (os.path.dirname(ag.fn_model)))
        sys.exit(-1)
    else:
        print('Using %s as a reference\n' % lst[0])
    fn_ref = lst[0]
else:
    fn_ref = ag.fn_ref

on = on_model.OpticNerveFit()
on.load(ag.fn_model)
on_cnt_frame_r, on_cnt_frame_l = on.make_center_frames()

img = nib.load(fn_ref)

on_cnt = on.make_center_0(shape_new=img.shape[:3])

hdr = img.header
hdr.set_data_dtype(np.int8)

dat_out = nib.Nifti1Image(on_cnt, img.affine, hdr)
nib.save(dat_out, ag.fn_out)

hdr.set_data_dtype(np.float)

dat_out = nib.Nifti1Image(on_cnt_frame_r, img.affine, hdr)
nib.save(dat_out, ag.fn_frame_out + '_r.nii.gz')

dat_out = nib.Nifti1Image(on_cnt_frame_l, img.affine, hdr)
nib.save(dat_out, ag.fn_frame_out + '_l.nii.gz')

