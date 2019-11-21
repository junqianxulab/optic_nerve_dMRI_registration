#!/usr/bin/env python

from on_utils import run_command, filename_wo_ext, fn_from_ls, save_list
import sys
import os
import nibabel as nib

fn_output_txt = '_Rigid.txt'

if len(sys.argv) < 2:
    sys.stderr.write('Usage: %s filename_DWI.txt [PE="PA"]\n')
    sys.exit(-1)
else:
    fn_dwi_txt = sys.argv[1]

if len(sys.argv) > 2:
    middle = sys.argv[2]
else:
    middle = 'PA'

subject = os.path.basename(fn_dwi_txt).split('_')
if len(subject) == 0:
    subject = ''
    prefix = ''
else:
    #subject = subject[0]
    subject = '_'.join(subject[:-1])
    prefix = subject + '_'

dn_base = os.path.dirname(fn_dwi_txt)

with open(fn_dwi_txt) as f:
    lst_dwi = [tmp.strip() for tmp in f.readlines() if len(tmp)>1]

lst_nonfile = [filename for filename in lst_dwi if not os.path.isfile(os.path.join(dn_base, filename))]
if lst_nonfile:
    print 'file not exist: ', lst_nonfile
    os.path.exit(-1)
if False:
    for i in range(len(lst_dwi)):
        filename = lst_dwi[i]
        if os.path.dirname(filename) != '' and os.path.dirname(filename) != '.' and os.path.dirname(filename) != './':
            if '0' <= os.path.basename(filename)[0] <= '9':
                lst_dwi[i] = prefix+os.path.basename(filename)
            elif '0' <= os.path.dirname(filename).split('/')[-1][0] <= '9':
                lst_dwi[i] = '%s%s_%s_%s' % (prefix, os.path.dirname(filename).split('/')[-1].split('_')[0], middle, os.path.basename(filename))
            else:
                lst_dwi[i] = '%sordered%s_%s_%s' % (prefix, i+1, middle, os.path.basename(filename))

lst_b0 = []
lst_fn = []
lst_nframe = []
for i in range(len(lst_dwi)):
    fn_dwi = os.path.join(dn_base, lst_dwi[i])
    lst_fn.append(fn_dwi)
    lst_nframe.append(nib.load(fn_dwi).shape[-1])
    fn_dwi_wo_ext = filename_wo_ext(fn_dwi)
    bn_dwi = os.path.basename(fn_dwi_wo_ext)
    run_command('extract_b0_dwi_mean.py %s %s.bval %s' % (fn_dwi, fn_dwi_wo_ext, os.path.basename(fn_dwi)))
    lst_b0.append('%s_b0' % bn_dwi)

if False:
    subject = lst_b0[0].split('_')
    is_number = False
    for i in range(len(subject)):
        try:
            int(subject[i])
        except:
            pass
        else:
            is_number = True
        if is_number:
            break
    subject = '_'.join(subject[:i])

print subject
print lst_b0

fn_output_txt = prefix + fn_output_txt
bn_mean = '%sb0mean' % prefix
#identmat = '$FSLDIR/etc/flirtsch/ident.mat'
fn_tempmat = 'tmp.mat'
fn_mask = '%sb0_mask.nii.gz' % prefix
nitr = 5

if not os.path.isfile(fn_mask):
    sys.stderr.write('mask file is not exist. run the following with proper thr value:\n')
    sys.stderr.write('thr=200 ; \ \n')
    lst_tmp = ['', 'tmp1', 'tmp2']
    ind_tmp = 1
    sys.stderr.write('fslmaths %s -thr $thr -bin %s ; \ \n' % (lst_b0[0], lst_tmp[ind_tmp]))
    for filename in lst_b0[1:-1]:
        sys.stderr.write('fslmaths %s -thr $thr -bin -add %s %s; \ \n' % (filename, lst_tmp[ind_tmp], lst_tmp[ind_tmp*(-1)]))
        ind_tmp *= (-1)
    sys.stderr.write('fslmaths %s -thr $thr -bin -add %s %s \n' % (lst_b0[-1], lst_tmp[ind_tmp], fn_mask))
    sys.exit(-1)

def init(lst_b0):
    for i in range(len(lst_b0)):
        mat = '%s_to_mean.mat' % lst_b0[i]
        #run_command('cp %s %s' % (identmat, mat))
        run_command('cp %s.nii.gz %s_xenc.nii.gz' % (lst_b0[i], lst_b0[i]))

def calc(lst_b0):
    run_command('fslmerge -t %s_concat %s' % (bn_mean, ' '.join(['%s_xenc' % tmp for tmp in (lst_b0)])))
    run_command('fslmaths %s_concat -Tmean %s' % (bn_mean, bn_mean))

def xalign(lst_b0, t):
    for i in range(len(lst_b0)):
        mat = '%s_to_mean.mat' % lst_b0[i]

        cmd = 'antsRegistration -d 3 \
                -n BSpline[3] \
                --convergence 10 \
                --transform Rigid[3] \
                --smoothing-sigmas 0mm \
                --metric MI[%s.nii.gz,%s_xenc.nii.gz,1,16] \
                -o [%s,%s_xenc.nii.gz] \
                --masks [%s,%s] \
                --shrink-factors 1 \
                ' % (bn_mean, lst_b0[i], fn_tempmat, lst_b0[i], fn_mask, fn_mask)
        run_command(cmd)

        if t == 0:
            run_command('cp %s0GenericAffine.mat %s0GenericAffine.mat' % (fn_tempmat, mat))
        else:
            cmd = 'ComposeMultiTransform 3 %s0GenericAffine.mat %s0GenericAffine.mat %s0GenericAffine.mat' \
                    % (mat, fn_tempmat, mat)
                    #% (mat, mat, fn_tempmat)
            run_command(cmd)

def align(lst_b0):
    '''
    align(param, mode, slc, seg=None)
    mode: 'b0', 'seggeom, 'segdwi'
    If mode is 'segdwi', set seg.
    '''
    init(lst_b0)
    calc(lst_b0)
    
    for t in range(nitr):
        xalign(lst_b0, t)
        calc(lst_b0)

def applywarp(lst_b0, lst_fn):
    lst_fnout = []
    for i in range(len(lst_b0)):
        mat = '%s_to_mean.mat0GenericAffine.mat' % lst_b0[i]
        f_in = lst_fn[i]
        f_out = filename_wo_ext(os.path.basename(lst_fn[i])) + '_rigid.nii.gz'
        cmd = 'antsApplyTransforms -d 3 -e 3 -i %s -r %s.nii.gz -o %s -n BSpline[3] -t %s' \
                % (f_in, lst_b0[i], f_out, mat)
        run_command(cmd)
        lst_fnout.append(f_out)
    return lst_fnout

    
with open(fn_output_txt, 'w') as fout:
    fout.write('%s\n' % ' '.join( ['%s_to_mean.mat' % tmp for tmp in lst_b0] ) )
    fout.write('%s\n' % ' '.join( [str(tmp) for tmp in lst_nframe] ) )

align(lst_b0)
lst_fnout = applywarp(lst_b0, lst_fn)

print('')
print('Check registration result (scroll volumes):')
print('    fslview %s_concat' % (bn_mean))
print('')
print('If registration is good, merge the registered DWI files:')
print('    fslmerge -t %s %s' % ('%s%s_%smerged_rigid' % (prefix, middle, len(lst_fnout)), ' '.join(lst_fnout)))

