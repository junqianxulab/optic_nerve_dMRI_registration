#!/usr/bin/env python

# coding: utf-8
# # ON DWI centerline extraction

# ### import modules
import numpy as np
import os
import sys
from on_utils import filename_wo_ext
from on_dbsi_utils import run_command
from on_dbsi_utils import reg_ms
import argparse

debug = True
verbose = False

#working_dir = '/Users/joowon/research/optic_nerve/DBSI_ON/DBSIONE008/16_0128/preproc_12_15/DBSIONE008_reg_wo_eddy'
working_dir = '.'

#filename = on_dbsi_utils.get_dwi_filename(working_dir)
#filename = 'DBSIONE008_PA_2merged_rigid.nii.gz'

parser = argparse.ArgumentParser()
parser.add_argument('fn_dwi', help='dwi')
parser.add_argument('-i', help='initial drawing', dest='fn_init', default=None)
parser.add_argument('-c', help='centerline', dest='fn_cent', default=None)
parser.add_argument('-e', help='exclusion', dest='fn_exclusion', default=None)
parser.add_argument('--dxy', help='search_dxy', dest='dxy', type=float, default=2.0)

ag = parser.parse_args()
filename = ag.fn_dwi
if not os.path.isfile(filename):
    sys.stderr.write('%s not exist.\n' % filename)
    sys.exit(-1)

if ag.fn_init is None:
    filename_on_init = filename_wo_ext(filename) + '_on_init.nii.gz'
else:
    filename_on_init = ag.fn_init

if ag.fn_cent is None:
    filename_centerline = filename_wo_ext(filename) + '_on_centerline.nii.gz'
else:
    filename_centerline = ag.fn_cent

if ag.fn_exclusion is None:
    filename_exclusion = None
    filename_excluded = filename
else:
    filename_exclusion = ag.fn_exclusion
    filename_excluded = filename_wo_ext(filename) + '_excluded.nii.gz'
    filename_exclusion_mask = filename_wo_ext(filename) + '_exclusion_mask.nii.gz'
    run_command('fslmaths %s -binv %s' % (filename_exclusion, filename_exclusion_mask))
    run_command('fslmaths %s -mas %s %s' % (filename, filename_exclusion_mask, filename_excluded))

if False:
    if len(sys.argv) < 2:
        sys.stderr.write('Usage: %s dwi_filename [centerline_filename]\n' % os.path.basename(sys.argv[0]))
        sys.exit(-1)

    filename = sys.argv[1]
    if len(sys.argv) > 2:
        filename_centerline = sys.argv[2]
    else:
        filename_centerline = filename_wo_ext(filename) + '_on_centerline.nii.gz'

filename_on_init = filename_wo_ext(filename) + '_on_init.nii.gz'
filename_flt_1 = filename_wo_ext(filename) + '_fltered_1.nii.gz'
filename_flt = filename_wo_ext(filename) + '_fltered.nii.gz'
filename_edge = filename_wo_ext(filename) + '_fltered_edge.nii.gz'
filename_centerline_dilated = filename_wo_ext(filename) + '_on_centerline_dilated.nii.gz'
filename_model = filename_wo_ext(filename) + '_on_model.nii.gz'
filename_model_1st_bin = filename_wo_ext(filename) + '_on_model_1st_bin.nii.gz'
filename_on_object = filename_wo_ext(filename) + '_on_model'

if False:
    if len(sys.argv) > 3:
        filename_exclusion = sys.argv[3]
        filename_excluded = filename_wo_ext(filename) + '_excluded.nii.gz'
        filename_exclusion_mask = filename_wo_ext(filename) + '_exclusion_mask.nii.gz'
        run_command('fslmaths %s -binv %s' % (filename_exclusion, filename_exclusion_mask))
        run_command('fslmaths %s -mas %s %s' % (filename, filename_exclusion_mask, filename_excluded))
    else:
        filename_exclusion = None
        filename_excluded = filename


if not os.path.isfile(filename_centerline):
    import on_centerline
    on_centerline.run(os.path.join(working_dir, filename_excluded),
                      os.path.join(working_dir, filename_on_init),
                      os.path.join(working_dir, filename_flt_1),
                      os.path.join(working_dir, filename_centerline),
                      os.path.join(working_dir, filename_centerline_dilated),)
    # ## Check and modify centerline
    #run_command('fslview %s %s -l Red' % (os.path.join(working_dir, filename), os.path.join(working_dir, filename_centerline)))
    print 'fslview %s %s -l Red' % (os.path.join(working_dir, filename), os.path.join(working_dir, filename_centerline))
    #sys.exit(0)

# ## Fitting
import nibabel as nib
import on_model

print 'Optic nerve nonlinear registration'

on = on_model.OpticNerveFit()
#on.read_dat(os.path.join(working_dir, filename_excluded))
on.read_dat(os.path.join(working_dir, filename))
on.read_init_center(os.path.join(working_dir, filename_centerline))

if filename_exclusion is not None:
    on.set_exclusion(filename_exclusion)

print 'done: reading initial centerline'
on.fit(dxy=ag.dxy)
#on.fit_simple()
print 'done: fitting'

if debug:
    on.save(os.path.join(working_dir, 'debug_on_save_1_fit'))

on.outlier()
on.estimate()
print 'done: estimating outliers'

on.save(os.path.join(working_dir, filename_on_object))

on_dil = on.make_segmentation(gaussian_output=True)


on.hdr.set_data_dtype(on_dil.dtype)
img_tmp = nib.Nifti1Image(on_dil, on.aff, on.hdr)
nib.save(img_tmp, os.path.join(working_dir, filename_model))

#if filename_exclusion is None:
run_command('fslmaths %s -thr 1.2 -Tmax -bin %s' % (filename_model, filename_model_1st_bin))

print 'done: saving model'

# ## Registration

fn_in = os.path.join(working_dir, filename)
fn_model = os.path.join(working_dir, filename_model)

dn_reg = os.path.join(working_dir, 'registration')
fn_out_base = os.path.join(dn_reg, os.path.basename(filename_wo_ext(filename)))

run_command('mkdir -p %s' % dn_reg)

#fslroi
for f in range(on.shape[-1]):
    fn_frame = '%s_on_model_%03d.nii.gz' % (fn_out_base, f)
    cmd = 'fslroi %s %s %s 1' % (fn_model, fn_frame, f)
    run_command(cmd, verbose)

    fn_frame = '%s_frame_%03d.nii.gz' % (fn_out_base, f)
    cmd = 'fslroi %s %s %s 1' % (fn_in, fn_frame, f)
    run_command(cmd, verbose)

#fnirt?
#ants?
fn_ref_on_model = '%s_on_model_%03d.nii.gz' % (fn_out_base, 0)
fn_ref = '%s_on_model_%03d.nii.gz' % (fn_out_base, 0)
run_command('cp -f %s_frame_%03d.nii.gz %s_frame_%03d_xenc.nii.gz' % (fn_out_base, 0, fn_out_base, 0), verbose)
run_command('cp -f %s_on_model_%03d.nii.gz %s_on_model_%03d_xenc.nii.gz' % (fn_out_base, 0, fn_out_base, 0), verbose)

for f in range(1, on.shape[-1]):
    fn_frame = '%s_on_model_%03d.nii.gz' % (fn_out_base, f)
    fn_warp = '%s_on_model_%03d_warp' % (fn_out_base, f)
    fn_out = '%s_on_model_%03d_xenc.nii.gz' % (fn_out_base, f)
    cmd = reg_ms(fn_ref_on_model, fn_frame, fn_warp, fn_out, fn_mask=None)
    run_command(cmd, verbose)

#applywarp
    fn_frame = '%s_frame_%03d.nii.gz' % (fn_out_base, f)
    fn_out = '%s_frame_%03d_xenc.nii.gz' % (fn_out_base, f)
    cmd = 'antsApplyTransforms -d 3 -i %s -r %s -o %s -n BSpline[3] -t %s1Warp.nii.gz'         % (fn_frame, fn_ref, fn_out, fn_warp)
    run_command(cmd, verbose)


# ### Merge registered frames

fn_reg = '%s_nonlin_reg.nii.gz' % fn_out_base
fns_frame = ['%s_frame_%03d_xenc.nii.gz' % (fn_out_base, f) for f in range(on.shape[-1])]
cmd = 'fslmerge -t %s %s' % (fn_reg, ' '.join(fns_frame))
run_command(cmd, verbose)
run_command('cp %s ./' % fn_reg, verbose)

# ### Merge registered centerlines (optional)

fn_model_reg = '%s_on_model_reg.nii.gz' % fn_out_base
fns_frame = ['%s_on_model_%03d_xenc.nii.gz' % (fn_out_base, f) for f in range(on.shape[-1])]
cmd = 'fslmerge -t %s %s' % (fn_model_reg, ' '.join(fns_frame))
run_command(cmd, verbose)
run_command('cp %s ./' % fn_model_reg, verbose)

print 'done: registration'
print ''
print 'to create optic nerve center estimation, run:'
print '    on_create_center_from_model.py -r %s %s' % (os.path.basename(fn_reg), os.path.basename(filename_on_object))
print ''
print 'Check the result:'
print '    fslview %s' % fn_reg

