#!/usr/bin/env python

# ON DWI centerline extraction and nonlinear registration

import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
import os
import sys
import on_dbsi_utils

# USER INPUT: directory / filename

def run(filename, filename_on_init, filename_flt, filename_centerline, filename_centerline_dilated, sigma=1, method_init=2, method_modify=1, nitr=2):
    # Load image file
    img = nib.load(filename)
    dat = img.get_data()
    hdr = img.get_header()

    dat_init = nib.load(filename_on_init).get_data()
    p_1 = zip(*(dat_init == 1).nonzero())
    p_2 = zip(*(dat_init == 2).nonzero())

    if len((dat_init==1).nonzero()[0]) == 0:
        sys.stderr.write('right nerve root (1) is not set.\n')
        sys.exit(-1)

    if len((dat_init==2).nonzero()[0]) == 0:
        sys.stderr.write('left nerve root (2) is not set.\n')
        sys.exit(-1)
        
    if len((dat_init==4).nonzero()[0]) < 2:
        sys.stderr.write('Not enough eyeball values (4) in on_init file.\n')
        sys.exit(-1)


    if len(p_1) == 0:
        print 'no points having value 1'
    elif len(p_1) > 1:
        print 'multiple points having value 1'
    if len(p_2) == 0:
        print 'no points having value 2'
    elif len(p_2) > 1:
        print 'multiple points having value 2'
        
    # Verify starting points

    print 'Point 1: ', p_1
    print 'Point 2: ', p_2


    # Eye ball intensity in a mask file
    VALUE_EYEBALL = -1

    # Apply Gaussian filter
    dat_flt = dat.copy()
    for frame in range(dat_flt.shape[-1]):
        dat_flt[:,:,:,frame] = ndimage.gaussian_filter(dat_flt[:,:,:,frame], sigma=sigma)

    dat_flt_ori = dat_flt.copy()

    # Save filtered image
    out_dtype = dat_flt.dtype
    hdr.set_data_dtype(out_dtype)
    img_flt = nib.Nifti1Image(dat_flt, img.get_affine(), hdr)
    nib.save(img_flt, filename_flt)

    # Apply mask
    dat_flt[ dat_flt < 0 ] = 0
    dat_flt[ dat_init == 4 ] = VALUE_EYEBALL

    # for debug
    if False:
        imin = 0
        imax = 200
        import optic_nerve
        optic_nerve.show_slice(dat_flt[:,:,:,3], k=5, imin=imin, imax=imax)

    # Initiate output centerline image
    out_dtype = np.int8
    dat_centerline = np.zeros(dat_flt.shape, dtype=out_dtype)

    # Find centerline in each frame
    num_dilation = 8
    nitr = 1

    if method_init == 1:
        shortest_path = lambda _point,_dat_3d:on_dbsi_utils.shortest_path_from_pts_to_intensity_dijk(_point, VALUE_EYEBALL, _dat_3d, min_thr=10)
    elif method_init == 2:
        shortest_path = lambda _point,_dat_3d:on_dbsi_utils.shortest_path_from_pts_to_intensity_dijk(_point, VALUE_EYEBALL, _dat_3d, min_thr=10,
                                            dist_diff=lambda x,y:(5.0*(np.abs(x-y)+0)/(x+y)+100.0/x+100.0/y))
    else:
        raise

    if method_modify == 1:
        modify_path = lambda _dat_3d,_path,_nitr:on_dbsi_utils.coms_near_xz_nbd(_dat_3d, _path, _nitr)
    elif method_modify == 2:
        modify_path = lambda _dat_3d,_path,_nitr:on_dbsi_utils.weighted_mean_near_xz_nbd(_dat_3d, _path, nitr=_nitr)
    else:
        raise

    for p in [p_1, p_2]:
        for frame in range(dat_flt.shape[-1]):
            path = shortest_path(p, dat_flt[:,:,:,frame])
            path_trim = on_dbsi_utils.trim(path[:-1])
            path_trim = modify_path(dat_flt[:,:,:,frame], path_trim, nitr)

            # dilate center line on xz plane
            nbds = on_dbsi_utils.dilate_to_xz(dat_flt[:,:,:,frame], 1, path_trim)
            # mark centerline
            for point in nbds:
                dat_centerline[point+(frame,)] = 2

            for point in path_trim:
                dat_centerline[point+(frame,)] = 1

    # Save centerline
    hdr.set_data_dtype(out_dtype)
    img_centerline = nib.Nifti1Image(dat_centerline==1, img.get_affine(), hdr)
    nib.save(img_centerline, filename_centerline)

    img_centerline = nib.Nifti1Image(dat_centerline>0, img.get_affine(), hdr)
    nib.save(img_centerline, filename_centerline_dilated)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write('Usage: %s filename [mask_filename=filename_on_init.nii.gz [exclusion_mask=None [sigma=1] ] ]\n' % sys.argv[0])
        sys.stderr.write('  filename_on_init.nii.gz:\n')
        sys.stderr.write('    1: right optic nerve root\n')
        sys.stderr.write('    2: left optic nerve root\n')
        sys.stderr.write('    4: eyeball\n')
        sys.exit(-1)

    filename = sys.argv[1]
    if len(sys.argv) > 2:
        filename_on_init = sys.argv[2]
    else:
        filename_on_init = on_dbsi_utils.filename_wo_ext(filename) + '_on_init.nii.gz'

    filename_flt = on_dbsi_utils.filename_wo_ext(filename) + '_fltered.nii.gz'
    filename_centerline = on_dbsi_utils.filename_wo_ext(filename) + '_on_centerline.nii.gz'
    filename_centerline_dilated = on_dbsi_utils.filename_wo_ext(filename) + '_on_centerline_dilated.nii.gz'
    #filename_centerline_sngl = on_dbsi_utils.filename_wo_ext(filename) + '_on_centerline_sngl.nii.gz'

    if len(sys.argv) > 3:
        filename_exclusion_mask = on_dbsi_utils.filename_wo_ext(filename) + '_exclusion_mask.nii.gz'
        filename_excluded = on_dbsi_utils.filename_wo_ext(filename) + '_excluded.nii.gz'
        on_dbsi_utils.run_command('fslmaths %s -binv %s' % (sys.argv[3], filename_exclusion_mask))
        on_dbsi_utils.run_command('fslmaths %s -mas %s %s' % (filename, filename_exclusion_mask, filename_excluded))
    else:
        filename_excluded = filename

    if len(sys.argv) > 4:
        sigma = float(sys.argv[4])
    else:
        sigma = 1
    print 'sigma = ', sigma

    run(filename=filename_excluded, filename_on_init=filename_on_init, filename_flt=filename_flt, filename_centerline=filename_centerline, filename_centerline_dilated=filename_centerline_dilated, sigma=sigma)

    print '## After checking and modifying %s, run on_dilate.py and on_nonlinear_reg.py' % (filename_centerline)
    print
    print 'fslview %s %s -l Yellow %s -l Red' % (filename, filename_centerline_dilated, filename_centerline)
    print 'on_dilate.py %s %s' % (filename, filename_centerline)
    print 'on_nonlinear_reg.py %s %s' % (filename, filename_centerline)
    print

