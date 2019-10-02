#!/usr/bin/env python
#
# coding: utf-8

# Optic nerve single slice dMRI registratino
#
# Joo-won Kim
# Icahn School of Medicine at Mount Sinai
#
# https://github.com/junqianxulab/optic_nerve_single_slice_dMRI_registration

import nibabel as nib
import numpy as np
import os
import sys
import scipy.ndimage
import scipy.interpolate
import argparse

def resize_img(dat, mask, d=5):
    nonzero = mask.nonzero()
    x0, xn = nonzero[0].min()-d, nonzero[0].max()+d
    y0, yn = nonzero[1].min()-d, nonzero[1].max()+d - 1
    if x0 < 0:
        x0 = 0
    if xn >= dat.shape[0]:
        xn = dat.shape[0] - 1
    if y0 < 0:
        y0 = 0
    if yn >= dat.shape[1]:
        yn = dat.shape[1] - 1

    return dat[x0:xn+1, y0:yn+1, :, :].copy(), mask[x0:xn+1, y0:yn+1].copy(), (x0, y0)


parser = argparse.ArgumentParser()

parser.add_argument('filename', help='nifti image')
parser.add_argument('-m', dest='mask', help='mask image', default=None)
parser.add_argument('-s', dest='size', help='Window size (voxel)', default=3, type=int)
parser.add_argument('--outsize', dest='size_out', help='output size (voxel)', default=5, type=int)
parser.add_argument('--sigma', dest='sigma', help='Gaussian sigma', default=1.0, type=float)
parser.add_argument('--resample', dest='resample', help='Resample ratio', default=1.0, type=float)

ag = parser.parse_args()

# set filenames
fn = ag.filename
if fn[-7:] == '.nii.gz':
    bn = fn[:-7]
elif fn[-4:] == '.nii':
    bn = fn[:-4]
else:
    bn = fn

if ag.mask is not None:
    fn_mask = ag.mask
else:
    fn_mask = bn + '_mask.nii.gz'
    if not os.path.isfile(fn_mask):
        fn_mask = bn + '_mask.nii'
        if not os.path.isfile(fn_mask):
            sys.stderr.write('mask file not found\n')
            sys.exit(-1)

size = ((ag.size-1)/2.0, (ag.size-1)/2.0)
size_out = (ag.size_out, ag.size_out)

img = nib.load(fn)
zoom_ori = img.header.get_zooms()

dat_ori_multi = img.get_data()
mask_ori_multi = nib.load(fn_mask).get_data()

def register_single_slice(dat_ori, mask_ori, ag, size, size_out, img, zoom_ori, bn):
    dat, mask, (x0, y0) = resize_img(dat_ori, mask_ori)

    search_domain = [ (xx, yy) for xx in np.arange(size[0], dat.shape[0]-size[0]+0.01, 0.2)
                               for yy in np.arange(size[1], dat.shape[1]-size[1]+0.01, 0.2) ]

    dat_mod = dat.astype(np.float)
    dat_max = np.zeros(dat.shape, dtype=dat.dtype)
    dat_reduced = np.zeros(size_out +(1, dat.shape[-1]), dtype=dat.dtype)
    d_size_out = ((size_out[0]-1)/2.0/ag.resample, (size_out[1]-1)/2.0/ag.resample)

    loc_x = np.zeros(dat_mod.shape[-1], dtype=np.float)
    loc_y = np.zeros(dat_mod.shape[-1], dtype=np.float)

    for f in range(dat_mod.shape[-1]):
        dat_mod[:,:,:,f][mask == 0] = 0
        dat_mod[:,:,:,f] = scipy.ndimage.gaussian_filter(dat_mod[:,:,:,f], sigma=ag.sigma)
        dat_mod[:,:,:,f] /= dat_mod[:,:,:,f].max()
        dat_mod[:,:,:,f] *= 100

        f_interp = scipy.interpolate.interp2d(range(dat_mod.shape[1]), range(dat_mod.shape[0]),
                                              dat_mod[:,:,0,f], kind='cubic', fill_value=0.0)

        # use optimization?
        max_sum = 0.0
        max_ind = -1
        for i, (xx, yy) in enumerate(search_domain):
            sum_i = f_interp(
                             np.arange(yy-size[1], yy+size[1]+0.1, 1.0),
                             np.arange(xx-size[0], xx+size[0]+0.1, 1.0)
            )
            if sum_i.sum() > max_sum:
                max_sum = sum_i.sum()
                max_ind = i
        imax = search_domain[max_ind]

        f_interp = scipy.interpolate.interp2d(range(dat.shape[1]), range(dat.shape[0]),
                                              dat[:,:,0,f], kind='cubic')
        xx, yy = imax
        loc_x[f] = xx
        loc_x[f] = yy

        dat_reduced[:, :, 0, f] = f_interp(
                                           np.linspace(yy-d_size_out[1], yy+d_size_out[1], size_out[1]),
                                           np.linspace(xx-d_size_out[0], xx+d_size_out[0], size_out[0])
                                           #np.arange(yy-(size_out[1]-1)/2.0, yy+(size_out[1]-1)/2.0+0.1, 1.0),
                                           #np.arange(xx-(size_out[0]-1)/2.0, xx+(size_out[0]-1)/2.0+0.1, 1.0)
        )
        for dx, dy in [ (t1, t2) for t1 in (-1, 0, 1) for t2 in (-1, 0, 1)]:
            dat_max[int(round(imax[0]))+dx, int(round(imax[1]))+dy, 0, f] = 1

    loc_x_rel = loc_x - round(np.median(loc_x))
    loc_y_rel = loc_y - round(np.median(loc_y))

    with open(bn + '_motion.csv', 'w') as fout:
        fout.write('frame,dx_voxel,dy_voxel,d_voxel,dx_mm,dy_mm,d_mm\n')
        for f in range(dat_mod.shape[-1]):
            fout.write('%d,%s,%s,%s,%s,%s,%s\n' % (f,
                    loc_x[f], loc_y[f], np.sqrt(loc_x[f]**2 + loc_y[f]**2),
                    loc_x_rel[f], loc_y_rel[f], np.sqrt(loc_x_rel[f]**2 + loc_y_rel[f]**2)
                    ))

    dat_mod_lg = np.zeros(dat_ori.shape, dtype=dat_mod.dtype)
    dat_mod_lg[x0:x0+dat_mod.shape[0],y0:y0+dat_mod.shape[1]] = dat_mod
    img_out = nib.Nifti1Image(dat_mod_lg, img.affine, img.header)
    nib.save(img_out, bn + '_gaussian.nii.gz')

    dat_max_lg = np.zeros(dat_ori.shape, dtype=dat_mod.dtype)
    dat_max_lg[x0:x0+dat_max.shape[0],y0:y0+dat_max.shape[1]] = dat_max
    img_out = nib.Nifti1Image(dat_max_lg, img.affine, img.header)
    nib.save(img_out, bn + '_rough_seg.nii.gz')

    zoom = list(zoom_ori)
    zoom[0] = zoom[0] / float(ag.resample)
    zoom[1] = zoom[1] / float(ag.resample)
    img.header.set_zooms(zoom)
    img_out = nib.Nifti1Image(dat_reduced, img.affine, img.header)
    nib.save(img_out, bn + '_reduced.nii.gz')

    #print 'fslview %s %s -l Red -t 0.3' % (fn, bn + '_rough_seg.nii.gz')
    print 'run dMRI fitting on %s' % (bn + '_reduced.nii.gz')

if len(mask_ori_multi.shape) == 2:
    mask_ori = mask_ori_multi
    dat_ori = dat_ori_multi
    register_single_slice(dat_ori, mask_ori, ag, size, size_out, img, zoom_ori, bn)
else:
    cmd_merge = 'fslmerge -z %s_rough_seg' % bn
    for s in range(mask_ori_multi.shape[2]):
        mask_ori = mask_ori_multi[:,:,s]
        dat_ori = dat_ori_multi[:,:,s:s+1,:]
        bn_sub = '%s_%s' % (bn, s)
        register_single_slice(dat_ori, mask_ori, ag, size, size_out, img, zoom_ori, bn=bn_sub)
        cmd_merge += str(' %s_rough_seg' % bn_sub)
    print(cmd_merge)

    

