#!/usr/bin/env python

from on_utils import run_command, filename_wo_ext, fn_from_ls, save_list
from on_utils import create_appa, create_acqparams, create_mask, total_readout_time
from on_utils import create_merged, create_index, create_mask, copy_dwi_files
import argparse
import os
import sys
import shutil
import nibabel as nib

#from on_parameter import create_parser_with_remainder
def create_parser_with_remainder():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='subject', default='')
    parser.add_argument('-b0', dest='lst_fn_b0', nargs='+')
    parser.add_argument('-dwi', dest='lst_fn_dwi', nargs='+')
    parser.add_argument('-thr', dest='thr', default=200)
    parser.add_argument('-o', dest='out_dir', default='./')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    return parser

if __name__ == '__main__':
    parser = create_parser_with_remainder()
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(-1)
    args = parser.parse_args()

    run_command('mkdir -p %s' % args.out_dir)
    prefix = args.out_dir
    if prefix != '' and prefix[-1] != '/':
        prefix += '/'
    if args.subject != '':
        prefix = '%s%s_' % (prefix, args.subject)
    
    fn_appa = prefix + 'APPA_b0.nii.gz'
    fn_appa_mask = prefix + 'APPA_b0_SNR_mask.nii.gz'
    fn_acqparams = prefix + 'acqparams.txt'
    fn_index = prefix + 'index.txt'

    fn_txt_b0_orig = prefix + 'APPA_b0.txt'
    fn_txt_dwi_orig = prefix + 'DWI_orig.txt'
    fn_txt_dwi = prefix + 'DWI.txt'

    if len(args.args) == 0:
        args.args.append('./')

    lst_fn_b0 = args.lst_fn_b0
    if lst_fn_b0 is None:
        lst_fn_b0 = fn_from_ls(args.args, check_file=fn_txt_b0_orig, contain='.nii', txt='Input file numbers to create an APPA b0 file (space-separated): ')
    save_list(fn_txt_b0_orig, lst_fn_b0)

    lst_fn_dwi = args.lst_fn_dwi
    if lst_fn_dwi is None:
        lst_fn_dwi = fn_from_ls(args.args, check_file=fn_txt_dwi_orig, contain='.nii', txt='Input file numbers to create a merged DWI file (space-separated): ')
    save_list(fn_txt_dwi_orig, lst_fn_dwi)

    _ =copy_dwi_files(lst_fn_b0, args.out_dir, prefix=prefix, middle='_b0Ref')
    create_appa(fn_appa, lst_fn_b0)
    create_acqparams(fn_acqparams, lst_fn_b0)
    create_mask(fn_appa, fn_appa_mask, thr=float(args.thr))

    has_pa, has_ap = create_index(fn_index, lst_fn_b0, lst_fn_dwi)
    if has_pa and (not has_ap):
        middle = '_PA'
    elif (not has_pa) and has_ap:
        middle = '_AP'
    else:
        middle = '_DWI'
    lst_fn_dwi_dest = copy_dwi_files(lst_fn_dwi, args.out_dir, prefix=prefix, middle=middle)
    save_list(fn_txt_dwi, lst_fn_dwi_dest)

    fn_dwi = '%s%s_%smerged' % (prefix, middle[1:], len(lst_fn_dwi))
    fn_bval, fn_bvec = create_merged(fn_dwi, lst_fn_dwi)

    fn_dwi_b0_mask = '%s_b0_mask.nii.gz' % fn_dwi
    run_command('extract_b0_dwi_mean.py %s.nii.gz %s %s' % (fn_dwi, fn_bval, os.path.basename(fn_dwi)))
    fn_dwi_b0 = '%s_b0.nii.gz' % fn_dwi
    create_mask(fn_dwi_b0, fn_dwi_b0_mask, thr=float(args.thr))

    print '###############################################################'
    print '# Check mask and apply mask.                                  #'
    print 'fslmaths %s -mas %s %s_masked' % (fn_appa, fn_appa_mask, filename_wo_ext(fn_appa))

