import os
from voxel_utils import *
from priodict import priorityDictionary
import numpy as np
import sys
import subprocess

def run_command(cmd, print_cmd = True):
    if print_cmd:
        print '>> %s' % cmd
        sys.stdout.flush()
    p = subprocess.call(cmd, shell=True)
    return p

def filename_wo_ext(filename):
    if filename[-7:] == '.nii.gz':
        return filename[:-7]
    elif filename[-4:] == '.nii':
        return filename[:-4]
    elif filename[-3:] == '.gz':
        return filename[:-3]
    elif filename[-5:] == '.bval':
        return filename[:-5]
    elif filename[-5:] == '.bvec':
        return filename[:-5]
    elif filename[-9:] == '.4dfp.img':
        return filename[:-9]
    elif filename[-5:] == '.4dfp':
        return filename[:-5]

    return filename

def save_img(dat, aff, hdr, filename):
    import nibabel as nib
    hdr.set_data_dtype(dat.dtype)
    img = nib.Nifti1Image(dat, aff, hdr)
    nib.save(img, filename)

def get_dwi_filename(directory):
    lstdir = os.listdir(directory)
    candi = [filename for filename in lstdir if 'eddy_unwarped.nii.gz' in filename]
    if len(candi) == 1:
        return candi[0]
    elif len(candi) > 1:
        len_candi = [len(filename) for filename in candi]
        filename = candi[np.argmin(len_candi)]
        print 'multiple dwi files. choosing shortest filename %s' % filename
        return filename
    
    candi = [filename for filename in lstdir if 'eddy_unwarp.nii.gz' in filename]
    if len(candi) == 1:
        print candi[0]
    elif len(candi) > 1:
        len_candi = [len(filename) for filename in candi]
        filename = candi[np.argmin(len_candi)]
        print 'multiple dwi files. choosing shortest filename %s' % filename
        return filename
    
    candi = [filename for filename in lstdir if 'rigid.nii.gz' in filename]
    if len(candi) == 1:
        print candi[0]
    elif len(candi) > 1:
        len_candi = [len(filename) for filename in candi]
        filename = candi[np.argmin(len_candi)]
        print 'multiple dwi files. choosing shortest filename %s' % filename
        return filename
    print 'dwi file not found. specify filename instead'
    return ''

def shortest_path_from_pts_to_intensity_dijk(point_0_s, point_1_int, dat, dist_func=lambda x:1.0/x, min_thr=0, max_thr=None, dist_diff=None, thr_distance=100000, zero_distance=None):
    '''
    find shortest path between point_0_s and points having intensity point_1_int in a 3d array dat.
    shortest_path_serial_dijk(point_0_s, point_1_int, dat, dist_func=lambda x:1.0/x, min_thr=0, max_thr=None, dist_diff=None, thr_distance=100000, zero_distance=None):
      point_0_s: list of starting points [(x, y, z), ...]
      point_1_int: ending point intensity intensity
      dat: numpy 3d array
      dist_func = lambda x:1.0/x: distance between a and b := euclidean_distance(a, b) * dist_func(b)
      min_thr = 0: ignore voxels having intensity < min_thr
      max_thr = 0: ignore voxels having intensity > max_thr
      dist_diff = None: if not None, distance between a and b := euclidean_distance(a, b) * dist_diff(a, b)
      thr_distance = 100000: the distance from/to the thresholded intensity voxels
      zero_distance = None: if not None, voxels having intensity zero_distance having distance 0 to their neighbors
    '''
    nx, ny, nz = dat.shape
    d1 = 1.0
    d2 = np.sqrt(2)
    d3 = np.sqrt(3)
    nbd_sets = [ (nbd_d1, d1), (nbd_d2, d2), (nbd_d3, d3) ]
    
    D = {}	# dictionary of final distances
    P = {}	# dictionary of predecessors
    Q = priorityDictionary()   # est.dist. of non-final vert.
    for point_0 in point_0_s:
        Q[point_0] = 0

    for v in Q:
        D[v] = Q[v]
        if dat[v] == point_1_int:
            point_1 = v
            break

        for nbd_d, dist_base in nbd_sets:
            for nbd in get_nbds_bdchk(v, nbd_d, nx, ny, nz):
                if dat[nbd] == point_1_int:
                    #print 'found point_2_int=%s' % point_1_int
                    #print len(Q)
                    distance = 0.01
                elif (zero_distance is not None and dat[nbd] == zero_distance):
                    distance = 0.01
                elif (min_thr is not None and dat[nbd] <= min_thr) or (max_thr is not None and max_thr <= dat[nbd]):
                    distance = thr_distance
                elif dist_diff is not None:
                    distance = dist_base * dist_diff(dat[nbd], dat[v])
                else:
                    distance = dist_base * dist_func(dat[nbd])
                if distance < 0.0:
                    print 'distance < 0'
                    return []

                vwLength = D[v] + distance
                
                if nbd in D:
                    if vwLength < D[nbd]:
                        raise ValueError, "Dijkstra: found better path to already-final vertex"
                elif nbd not in Q or vwLength < Q[nbd]:
                #if nbd not in Q or vwLength < Q[nbd]:
                    Q[nbd] = vwLength
                    P[nbd] = v
    Path = []
    end = point_1
    start = point_0_s
    while 1:
        Path.append(end)
        if end in start: break
        end = P[end]
        if len(Path) > 40:
            break
    Path.reverse()
    return Path

def trim(path):
    path_trim = path[:]
    def length(p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2])
    def length2(p1, p2):
        return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2
    i = 1
    while i < len(path_trim)-1:
        if length2(path_trim[i-1], path_trim[i+1]) < 4:
            path_trim.pop(i)
        else:
            i += 1
            
    return path_trim

def dilate_to_highest_nbd_itr_xz(dat, nitr, path):
    nbd_xz = [ (-1,0,0), (1,0,0), (0,0,-1), (0,0,1)]
    exclude = [point for point in path]
    bases = [[path[i]] for i in range(len(path))]
    for itr in range(nitr):
        for i in range(len(bases)):
            nbds = []
            for point in bases[i]:
                nbds += [nbd for nbd in get_nbds_bdchk(point, nbd_xz, *dat.shape) if nbd not in exclude]
            ints = [dat[nbd] for nbd in nbds]
            nbd_highest = nbds[np.argmax(ints)]
            if dat[nbd_highest] < 0:
                continue
            bases[i].append(nbd_highest)
            exclude.append(nbd_highest)
    total = []
    for base in bases:
        total += base
    return total

def dilate_to_xz(dat, nitr, path):
    nbd_xz = [ (-1,0,0), (1,0,0), (0,0,-1), (0,0,1),
               (-1,0,-1), (1,0,1), (1,0,-1), (-1,0,1)]
    bases = [[path[i]] for i in range(len(path))]
    for itr in range(nitr):
        for i in range(len(bases)):
            nbds = []
            for point in bases[i]:
                nbds += get_nbds_bdchk(point, nbd_xz, *dat.shape)
            bases[i] += list(set(nbds))
    total = []
    for base in bases:
        total += base
    return total

def coms_near_xz_nbd(dat_flt, voxels, nitr=5):
    # calulate com near nbds
    nbd_xz = [ (-1,0,0), (1,0,0), (0,0,-1), (0,0,1),
               (-1,0,-1), (1,0,1), (1,0,-1), (-1,0,1),
               (-2,0,0), (2,0,0), (0,0,-2), (0,0,2),
             ]

    com = voxels[:]
    for itr in range(nitr):
        changed = False

        for i in range(len(com)):
            nbds = [com[i]] + get_nbds_bdchk(com[i], nbd_xz, *dat_flt.shape)
            xc, yc, zc = center_of_mass(nbds, dat_flt)
            new_com = (int(np.round(xc)), int(np.round(yc)), int(np.round(zc)))
            if new_com != com[i]:
                com[i] = new_com
                changed = True
        i = 0
        while i < len(com)-1:
            if com[i] == com[i+1]:
                com.pop(i+1)
            else:
                i+=1

        if not changed:
            break
    return com

def reg_ms(fn_ref, fn_in, fn_warp, fn_out, fn_mask=None):
    cmd = '\
 antsRegistration -d 3 \
 -n BSpline[3] \
 --convergence 10 \
 --transform SyN[0.5,3,0] \
 --shrink-factors 1 \
 --smoothing-sigmas 0mm \
 --metric MeanSquares[%s,%s] \
 -r identity \
 -o [%s,%s] \
 ' % (fn_ref, fn_in, fn_warp, fn_out)
    if fn_mask is not None:
        cmd = '%s --masks [%s,%s]' % (cmd, fn_mask, fn_mask)
    return cmd

def weighted_mean_near_xz_nbd(dat_flt, voxels, nitr=5):
    # calulate com near nbds
    nbd_xz = [ (-1,0,0), (1,0,0), (0,0,-1), (0,0,1),
               (-1,0,-1), (1,0,1), (1,0,-1), (-1,0,1),
#               (-2,0,0), (2,0,0), (0,0,-2), (0,0,2),
             ]
    weight_xz = [ 1,
               1.0/2, 1.0/2, 1.0/2, 1.0/2,
               1.0/2.4, 1.0/2.4, 1.0/2.4, 1.0/2.4,
               1.0/3, 1.0/3, 1.0/3, 1.0/3,
             ]

    cnt = voxels[:]
    for itr in range(nitr):
        changed = False

        for i in range(len(cnt)):
            candidates = [cnt[i]] + get_nbds_bdchk(cnt[i], nbd_xz, *dat_flt.shape)
            candidates_value = []
            for j in range(len(candidates)):
                a_candi = candidates[j]
                nbds = [a_candi] + get_nbds_bdchk(a_candi, nbd_xz, *dat_flt.shape)
                candidates_value.append(sum([ dat_flt[nbds[k]]*weight_xz[k] for k in range(len(nbds))]))

            arg_max = np.argmax(candidates_value)
            #print i, arg_max, len(candidates_value)
            #print candidates_value
            if arg_max != 0:
                cnt[i] = candidates[arg_max]
                changed = True
        i = 0
        while i < len(cnt)-1:
            if cnt[i] == cnt[i+1]:
                cnt.pop(i+1)
            else:
                i+=1

        if not changed:
            break
    return cnt

def get_b0_from_bval(filename, threshold=90, return_verbose=False, verbose=True):
    fin = open(filename)
    frames = [int(value) for value in fin.readline().strip().split()]
    b0frames = []
    rtn_str = '# '
    for i in range(len(frames)):
        if frames[i] < threshold:
            b0frames.append(i)
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
        return b0frames, rtn_str
    return b0frames

def set_seg_frames(b0frames, nvolume):
    segframes = [ range(1+int(b0frames[j-1]), int(b0frames[j]))
            for j in range(1, len(b0frames)) if int(b0frames[j])-int(b0frames[j-1]) > 1]
    if int(b0frames[-1]) < int(nvolume)-1:
        segframes.append( range(1+int(b0frames[-1]), int(nvolume) ))

    return segframes

def twoD_Gaussian_raw((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g

def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta):
    return twoD_Gaussian_raw((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta).ravel()

def two_2d_Gaussian((x, y), amp1, amp2, xo, yo, sigma_x1, sigma_x2, sigma_y1, sigma_y2, theta):
    g1 = twoD_Gaussian((x, y), amp1, xo, yo, sigma_x1, sigma_y1, theta)
    g2 = twoD_Gaussian((x, y), amp1*np.exp(amp2), xo, yo, sigma_x2, sigma_y2, theta)
    return (-g1+g2).ravel()

def rotation(angle, (x, y)):
    cos_ = math.cos(angle)
    sin_ = math.sin(angle)
    return cos_*x - sin_*y, sin_*x + cos_*y

