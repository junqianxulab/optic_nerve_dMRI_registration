import nibabel as nib
import subprocess
import sys
import os
import numpy as np
import shutil

total_readout_time = 0.86 * (32-1) * 0.001

def run_command(cmd, print_cmd = True):
    if print_cmd:
        print '>> %s' % cmd
    p = subprocess.call(cmd, shell=True)
    return p

def filename_wo_ext(filename):
    if filename[-7:] == '.nii.gz':
        return filename[:-7]
    if filename[-4:] == '.nii':
        return filename[:-4]
    return filename

def read_lines(fn_txt_b0):
    if os.path.isfile(fn_txt_b0):
        with open(fn_txt_b0) as f:
            return [tmp.strip() for tmp in f.readlines() if tmp != '\n']
    return None

def fn_from_ls(directory='./', check_file=None, contain=None, txt=''):
    if check_file is not None:
        lst = read_lines(check_file)
        if lst is not None:
            print 'Data from %s:' % check_file
            print '\n'.join(lst)
            ans = raw_input('Use this data (Y/n)? ')
            if ans[0].lower() != 'n':
                return lst

    print ''
    print 'List of directory ', directory
    if type(directory) == type(''):
        dirs = [directory]
    elif type(directory) == type([]):
        dirs = directory
    else:
        dirs = ['./']

    lst = []
    for a_directory in directory:
        a_lst = os.listdir(a_directory)

        if a_directory != './':
            base = a_directory
            if base[-1] != '/':
                base += '/'
        else:
            base = ''
        #print a_directory, a_lst

        if contain is not None:
            a_lst = [base + tmp for tmp in a_lst if contain in tmp]
        else:
            a_lst = [base + tmp for tmp in a_lst].sort()
        a_lst.sort()
        lst += a_lst

    for i in range(len(lst)):
        print '%s: %s' % (i, lst[i])
    print
    lst_num = raw_input(txt)
    return [lst[int(tmp)] for tmp in lst_num.split()]

def save_list(filename, lst):
    if os.path.isfile(filename):
        run_command('cp -f %s %s.bak' % (filename, filename))
    with open(filename, 'w') as f:
        f.write('\n'.join([str(tmp) for tmp in lst]))

def copy_dwi_files(filelist, outdir='./', prefix='', middle=''):
    additional_filelist = []
    for filename in filelist:
        bval = filename_wo_ext(filename) + '.bval'
        bvec = filename_wo_ext(filename) + '.bvec'
        if os.path.isfile(bval) and os.path.isfile(bvec):
            additional_filelist += [bval, bvec]
    print additional_filelist

    to_copy = []
    count = 0
    for filename in filelist + additional_filelist:
        if os.path.abspath(os.path.dirname(filename)) == os.path.abspath(outdir):
            continue
        if '0' <= os.path.basename(filename)[0] <= '9':
            to_copy.append((filename, prefix+os.path.basename(filename)))
        elif '0' <= os.path.dirname(filename).split('/')[-1][0] <= '9':
            modified_filename = '%s%s%s_%s' % (prefix, os.path.dirname(filename).split('/')[-1].split('_')[0], middle, os.path.basename(filename))
            to_copy.append((filename, modified_filename))
        else:
            modified_filename = '%sordered%s%s_%s' % (prefix, count, middle, os.path.basename(filename))
    print to_copy
            
    for filename, to in to_copy:

        #if os.path.isfile(to):
        #    continue
        shutil.copy(filename, to)

    return [os.path.basename(tmp) for (_, tmp) in to_copy if '.nii' in tmp]

def create_appa(fn_appa, lst_fn):
    lst_fn_tmp = [ 'tmp_appa_%s' % i for i in range(len(lst_fn)) ]

    # extract 1st frame
    for i in range(len(lst_fn)):
        run_command('fslroi %s %s 0 1' % (lst_fn[i], lst_fn_tmp[i]))

    run_command('fslmerge -t %s %s' % (fn_appa, ' '.join(lst_fn_tmp)))
    for i in range(len(lst_fn)):
        run_command('rm -f %s.nii.gz 0 1' % lst_fn_tmp[i])

def create_acqparams(fn_acqparams, lst_fn, total_readout_time=total_readout_time):
    with open(fn_acqparams, 'w') as f:
        for filename in lst_fn:
            if 'pa' in filename.lower():
                f.write('0 1 0 %s\n' % total_readout_time)
                print '%s: PA' % filename
            elif 'ap' in filename.lower():
                f.write('0 -1 0 %s\n' % total_readout_time)
                print '%s: AP' % filename

def create_mask(fn_appa, fn_appa_mask, thr=200):
    from scipy import ndimage
    
    iteration_erosion=1
    iteration_dilation=2

    img = nib.load(fn_appa)
    dat = img.get_data()
    if len(dat.shape) == 3:
        shape = dat.shape + (1,)
    else:
        shape = dat.shape
    mask = np.zeros(shape, dtype=np.int8)
    mask[dat>thr] = 1
    mask_clean = np.zeros(shape, dtype=np.int8)

    for i in range(shape[3]):
        mask_i = mask[:,:,:,i]

        if iteration_dilation>0:
            inverse_i = np.logical_not(mask_i)
            inverse_clean_i = ndimage.binary_erosion(inverse_i)
            for j in range(iteration_dilation-1):
                inverse_clean_i = ndimage.binary_erosion(inverse_clean_i)
            for j in range(iteration_dilation):
                inverse_clean_i = ndimage.binary_propagation(inverse_clean_i, mask=inverse_i)
            mask_clean_i = np.logical_not(inverse_clean_i)
        else:
            mask_clean_i = mask_i.copy()

        if iteration_erosion>0:
            for j in range(iteration_erosion):
                mask_clean_i = ndimage.binary_erosion(mask_clean_i)
            for j in range(iteration_erosion):
                mask_clean_i = ndimage.binary_propagation(mask_clean_i, mask=mask_i)

        mask_clean[:,:,:,i] = mask_clean_i
    del mask_i, mask
    hdr = img.get_header()
    hdr.set_data_dtype(np.int8)

    if len(dat.shape) == 3:
        mask_clean = mask_clean[:,:,:,0]
    img_mask = nib.Nifti1Image(mask_clean, img.get_affine(), hdr)
    nib.save(img_mask, fn_appa_mask)
    del mask_clean, img_mask, dat, img

def create_merged(fn_dwi, lst_fn):
    run_command('fslmerge -t %s %s' % (fn_dwi, ' '.join(lst_fn)))
    lst_fn_base = [ filename_wo_ext(filename) for filename in lst_fn ]
    fn_dwi_base = filename_wo_ext(fn_dwi)
    fn_bval = '%s.bval' % fn_dwi_base
    fn_bvec = '%s.bvec' % fn_dwi_base
    run_command('paste %s.bval > %s' % ('.bval '.join(lst_fn_base), fn_bval))
    run_command('paste %s.bvec > %s' % ('.bvec '.join(lst_fn_base), fn_bvec))
    return fn_bval, fn_bvec

def create_index(fn_index, lst_fn_b0, lst_fn_dwi):
    index = []
    i_dwi = 0
    i_b0 = 0
    has_pa = ''
    has_ap = ''
    while i_b0 < len(lst_fn_b0) and i_dwi < len(lst_fn_dwi):
        # update direction for new i_dwi
        if 'pa' in lst_fn_dwi[i_dwi].lower():
            direction = 'PA'
            has_pa = 'PA'
        elif 'ap' in lst_fn_dwi[i_dwi].lower():
            direction = 'AP'
            has_ap = 'AP'
        else:
            direction = 'PA'
            has_pa = 'PA'

        if direction.lower() in lst_fn_b0[i_b0].lower():
            img = nib.load(lst_fn_dwi[i_dwi])
            i_b0 += 1
            i_dwi += 1
            index.append( ('%s ' % i_b0) * img.shape[3] )
            print '%s %s: %s' % (i_b0, direction, img.shape[3])

        else:
            i_b0 += 1

    with open(fn_index, 'w') as f:
        f.write('%s\n' % ''.join(index))
    return has_pa, has_ap

def create_topup_command(fn_in=None, fn_acqparams=None, fn_outtopup=None, fn_outimage=None, fn_outfield=None, fn_outlog=None, topup='topup', config='', subject='topup'):
    #>> topup_zer  --config=/home/xugroup/bin/b02b0_ON_1_3mm_iter8.cnf --datain=DBSIONV006_acqparams.txt --imain=DBSIONV006_AP_PA_SBRef.nii.gz --out=DBSIONV006_output_mb_sb_zer_i08/DBSIONV006_topup --iout=DBSIONV006_output_mb_sb_zer_i08/DBSIONV006_topup_warp --fout=DBSIONV006_output_mb_sb_zer_i08/DBSIONV006_topup_field --logout=DBSIONV006_output_mb_sb_zer_i08/DBSIONV006_topup_SBRef.log -v 1> DBSIONV006_output_mb_sb_zer_i08/DBSIONV006_topup_SBRef.log.out 2> DBSIONV006_output_mb_sb_zer_i08/DBSIONV006_topup_SBRef.log.err
    curr_vars = locals()
    errorlog = [variable for variable in curr_vars if curr_vars[variable] is None ]
    if len(errorlog) > 0:
        sys.stderr.write('missing parameter in create_topup_command()\n')
        sys.stderr.write('  %s\n' % (', '.join(errorlog)))
        return ''

    options = [topup,
            '--datain=%s' % fn_acqparams,
            '--imain=%s' % fn_in,
            '--out=%s' % fn_outtopup,
            '--iout=%s' % fn_outimage,
            '--fout=%s' % fn_outfield,
            '--logout=%s' % fn_outlog,
            ]
    if config is not '':
        options.append('--config=%s' % config)
    options += [
            '-v',
            '1> %s_log.out' % subject,
            '2> %s_log.err' % subject,
            ]

    return ' '.join(options)

def create_eddy_command(fn_in=None, fn_out=None, fn_bvals=None, fn_bvecs=None, fn_mask=None, fn_topup=None, fn_acqparams=None, fn_index=None, eddy='eddy', flm='quadratic', fwhm=0.0, subject='eddy'):
    #>> eddy --flm=quadratic --fwhm=0.0 --acqp=DBSIONV006_acqparams.txt --bvals=DBSIONV006_PA_2avg.bval --bvecs=DBSIONV006_PA_2avg.bvec --imain=DBSIONV006_PA_2avg.nii.gz --index=DBSIONV006_index.txt --mask=DBSIONV006_output_mb_sb_zer_i08/DBSIONV006_SNR_mask.nii.gz --topup=DBSIONV006_output_mb_sb_zer_i08/DBSIONV006_topup --out=/home/xugroup/DATA/OpticNerve/DBSI_ON/DBSIONV006/15_1211/preproc_sb/DBSIONV006_output_mb_sb_zer_i08/DBSIONV006_eddy_unwarped_images -v 1> DBSIONV006_output_mb_sb_zer_i08/DBSIONV006_eddy_unwarped_images.out 2> DBSIONV006_output_mb_sb_zer_i08/DBSIONV006_eddy_unwarped_images.err
    curr_vars = locals()
    errorlog = [variable for variable in curr_vars if curr_vars[variable] is None ]
    if len(errorlog) > 0:
        sys.stderr.write('missing parameter in create_eddy_command()\n')
        sys.stderr.write('  %s\n' % (', '.join(errorlog)))
        return ''

    options = [eddy,
            '--flm=%s' % flm,
            '--fwhm=%s' % fwhm,
            '--acqp=%s' % fn_acqparams,
            '--bvals=%s' % fn_bvals,
            '--bvecs=%s' % fn_bvecs,
            '--imain=%s' % fn_in,
            '--index=%s' % fn_index,
            '--mask=%s' % fn_mask,
            '--topup=%s' % fn_topup,
            '--out=%s' % fn_out,
            '-v',
            '1> %s_log.out' % subject,
            '2> %s_log.err' % subject,
            ]
    return ' '.join(options)

