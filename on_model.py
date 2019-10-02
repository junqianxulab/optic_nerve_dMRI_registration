#!/usr/bin/env python

# ON DWI centerline extraction and nonlinear registration

import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
import os
import sys
import on_dbsi_utils
import pickle
import time
import scipy.optimize as opt
import math
import scipy.interpolate

def twoD_Gaussian_raw((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) - 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g

def _twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta):
    return twoD_Gaussian_raw((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta).ravel()

# constraints: 0 < amp1 < 10, 0 < amp2 < 10, |xo - nx/2| < 3, |yo - ny/2| < 3, 0 < sigma, 0 <= theta < pi

def two_2d_Gaussian((x, y), amp1, amp2, xo, yo, sigma_x1, sigma_x2, sigma_y1, sigma_y2, theta):
    g1 = twoD_Gaussian_raw((x, y), amp1, xo, yo, sigma_x1, sigma_y1, theta)
    #g2 = twoD_Gaussian_raw((x, y), amp1*np.exp(amp2), xo, yo, sigma_x2, sigma_y2, theta)
    g2 = twoD_Gaussian_raw((x, y), amp2, xo, yo, sigma_x2, sigma_y2, theta)
    return (-g1+g2).ravel()

def rotation(angle, (x, y)):
    cos_ = math.cos(angle)
    sin_ = math.sin(angle)
    return cos_*x - sin_*y, sin_*x + cos_*y

def create_filter_box(size=5, radius_o=4, radius_i=3):
    filter_box = np.ones((2*size+1,2*size+1), dtype=float)
    for i in range(2*size+1):
        for j in range(2*size+1):
            if (size-i)**2 + (size-j)**2 > radius_o**2:
                filter_box[i,j] = 0.4
            elif (size-i)**2 + (size-j)**2 > radius_i**2:
                filter_box[i,j] = 0.7
    return filter_box

if False:
    _dxy = 1.5
    constraints = (
            {   'type': 'ineq',
                'fun': lambda x: np.array([x[2] + _dxy]),
                'jac': lambda x: np.array([0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
            },
            {   'type': 'ineq',
                'fun': lambda x: np.array([-x[2] + _dxy]),
                'jac': lambda x: np.array([0, 0, -1, 0, 0, 0, 0, 0, 0], dtype=float)
            },
            {   'type': 'ineq',
                'fun': lambda x: np.array([x[3] + _dxy]),
                'jac': lambda x: np.array([0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float)
            },
            {   'type': 'ineq',
                'fun': lambda x: np.array([-x[3] + _dxy]),
                'jac': lambda x: np.array([0, 0, 0, -1, 0, 0, 0, 0, 0], dtype=float)
            },
            {   'type': 'ineq',
                'fun': lambda x: np.array([x[0] - 0.01]),
                'jac': lambda x: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
            },
            {   'type': 'ineq',
                'fun': lambda x: np.array([-x[0] + 5]),
                'jac': lambda x: np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
            },
            {   'type': 'ineq',
                'fun': lambda x: np.array([x[1] - 0.01]),
                'jac': lambda x: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
            },
            {   'type': 'ineq',
                'fun': lambda x: np.array([-x[1] + 5]),
                'jac': lambda x: np.array([0, -1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
            },
            {   'type': 'ineq',
                'fun': lambda x: np.array([x[4] - 0.01]),
                'jac': lambda x: np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)
            },
            {   'type': 'ineq',
                'fun': lambda x: np.array([x[5] - 0.01]),
                'jac': lambda x: np.array([0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=float)
            },
            {   'type': 'ineq',
                'fun': lambda x: np.array([x[6] - 0.01]),
                'jac': lambda x: np.array([0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=float)
            },
            {   'type': 'ineq',
                'fun': lambda x: np.array([x[7] - 0.01]),
                'jac': lambda x: np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=float)
            },
    )

    bounds = (
            (0.01, 5),
            (0.01, 5),
            (-_dxy, _dxy),
            (-_dxy, _dxy),
            (0.01, None),
            (0.01, None),
            (0.01, None),
            (0.01, None),
            (None, None)
    )

def to_min_two_2d_Gaussian_base((amp1, amp2, xo, yo, sigma_x1, sigma_x2, sigma_y1, sigma_y2, theta), (x, y), (xn, yn), on_edge):
    return ((two_2d_Gaussian((x, y), amp1, amp2, xo+yn/2, yo+xn/2, sigma_x1, sigma_x2, sigma_y1, sigma_y2, theta) - on_edge.ravel())**2).sum()


class OpticNerveFit:
    def __init__(self, dat=None):
        self.dat = dat
        self.aff = None
        self.hdr = None
        self.init_center_r = None
        self.init_center_l = None
        self.on_mat_r = None
        self.on_mat_l = None
        self.outlier_r = None
        self.outlier_l = None
        self.dat_gaussian_1 = None
        self.dat_edge = None
        self.dat_exclusion = None
        self.centers = None
        self.coeffs = None
        self.num_seg = None
        self.rss_mean = None
        self.shape = None
        self.ind_x = 0
        self.ind_y = 1
        self.ind_z = 2
        self.ind_f = 3
        self.value_out = -100000
        self.__store__ = [self.dat, self.aff, self.hdr, self.init_center_r, self.init_center_l,
                self.on_mat_r, self.on_mat_l,
                self.outlier_r, self.outlier_l,
                self.dat_edge, self.centers, self.coeffs, self.num_seg, self.rss_mean, self.shape]

    def read_dat(self, filename):
        img = nib.load(filename)
        self.dat = img.get_data()
        self.hdr = img.get_header()
        self.aff = img.get_affine()
        self.shape = img.shape

    def set_exclusion(self, filename):
        self.dat_exclusion = nib.load(filename).get_data()
        self.dat_exclusion_dil = ndimage.binary_dilation(self.dat_exclusion)

    def edge_filter(self):
        dat_flt = self.dat.copy().astype(np.float)
        dat_edge = np.zeros(self.shape, dtype=np.float)
        
        if self.dat_exclusion is not None:
            ind_exclusions = zip(*self.dat_exclusion.nonzero())
        for frame in range(dat_edge.shape[-1]):
            dat_flt[:,:,:,frame] = ndimage.gaussian_filter(dat_flt[:,:,:,frame], sigma=0.5)

            if self.dat_exclusion is not None:
                # TODO: 
                # before sobel x/z, make exclusion mask as mean intensity of nbds
                # dx
                dat_flt_tmp = dat_flt[:,:,:,frame].copy()
                for ind in ind_exclusions:
                    nbd_inds = [(ind[0]+xx,ind[1],ind[2]) for xx in (-1, 1) if 0 <= ind[0]+xx < dat_flt_tmp.shape[0]]
                    nbd_values = [dat_flt[nbd[0], nbd[1], nbd[2], frame] for nbd in nbd_inds if self.dat_exclusion[nbd] == 0]
                    if len(nbd_values) > 0:
                        dat_flt_tmp[ind] = np.mean(nbd_values)
                    else:
                        dat_flt_tmp[ind] = np.mean([dat_flt[nbd[0], nbd[1], nbd[2], frame] for nbd in nbd_inds])
                dx = ndimage.sobel(dat_flt_tmp, 0)

                # dz
                dat_flt_tmp = dat_flt[:,:,:,frame].copy()
                for ind in ind_exclusions:
                    nbd_inds = [(ind[0],ind[1],ind[2]+xx) for xx in (-1, 1) if 0 <= ind[2]+xx < dat_flt_tmp.shape[2]]
                    nbd_values = [dat_flt[nbd[0], nbd[1], nbd[2], frame] for nbd in nbd_inds if self.dat_exclusion[nbd] == 0]
                    if len(nbd_values) > 0:
                        dat_flt_tmp[ind] = np.mean(nbd_values)
                    else:
                        dat_flt_tmp[ind] = np.mean([dat_flt[nbd[0], nbd[1], nbd[2], frame] for nbd in nbd_inds])
                dz = ndimage.sobel(dat_flt_tmp, 2)

            else:
                # sobel on coronal slices
                dx = ndimage.sobel(dat_flt[:,:,:,frame], 0, mode='nearest')
                dz = ndimage.sobel(dat_flt[:,:,:,frame], 2, mode='nearest')

            #if False:
            if self.dat_exclusion is not None:
                dx[self.dat_exclusion_dil>0] *= 0.5
                dz[self.dat_exclusion_dil>0] *= 0.5
                dx[self.dat_exclusion>0] = 0
                dz[self.dat_exclusion>0] = 0

            mag = np.hypot(dx, dz)
            
            # normalize
            mag *= 255.0 / mag.max()
            dat_edge[:,:,:,frame] = mag

        self.dat_edge = dat_edge

    def gaussian_filter(self, sigma=1.5):
        dat_flt = self.dat.copy()

        for frame in range(dat_flt.shape[-1]):
            dat_flt[:,:,:,frame] = ndimage.gaussian_filter(dat_flt[:,:,:,frame], sigma=sigma)

        self.dat_gaussian_1 = dat_flt


    def read_init_center(self, filename):
        dat_cnt = nib.load(filename).get_data()
        if dat_cnt.shape != self.shape:
            raise ValueError

        nx = self.shape[self.ind_x]
        nf = self.shape[self.ind_f]

        # find min/max y slices containing initial center voxel for all frames
        
        fzx = [self.ind_f, self.ind_z, self.ind_x]
        fzx.sort(reverse=True)

        tmp_y = dat_cnt[:nx/2,:,:,:].sum(fzx[0]).sum(fzx[1]).sum(fzx[2])
        ys_r = (tmp_y >= min(nf-2, np.median(tmp_y[tmp_y>0])-1)).nonzero()[0][[0, -1]]
        tmp_y = dat_cnt[nx/2:,:,:,:].sum(fzx[0]).sum(fzx[1]).sum(fzx[2])
        ys_l = (tmp_y >= min(nf-2, np.median(tmp_y[tmp_y>0])-1)).nonzero()[0][[0, -1]]

        on_mat_r = np.zeros(( nf, ys_r[1]-ys_r[0]+1, 5+9)) # 5: x, y, z, length, rss, 9: len(popt)
        on_mat_l = np.zeros(( nf, ys_l[1]-ys_l[0]+1, 5+9)) # 5: x, y, z, length, rss, 9: len(popt)

        # on_mat_lr: [frame, y, x/y/z/...]
        for dat_sub, ys_lr, on_mat_lr, dx in (
                (dat_cnt[:nx/2,:,:,:], ys_r, on_mat_r, 0),
                (dat_cnt[nx/2:,:,:,:], ys_l, on_mat_l, nx/2)
                ):
            dxy = np.zeros(3)
            #dxy[self.ind_y] = -ys_lr[0]
            dxy[self.ind_x] = dx
            for f in range(nf):
                center_points = (np.array(dat_sub[:,ys_lr[0]:ys_lr[1]+1,:,f].nonzero()).T + dxy)
                on_mat_lr[f, :, :3] = [self.value_out, self.value_out, self.value_out]
                for point in center_points:
                    on_mat_lr[f, int(point[1]), :3] = [point[0], point[1] + ys_lr[0], point[2]]

        self.init_center_r = ys_r
        self.init_center_l = ys_l
        self.on_mat_r = on_mat_r
        self.on_mat_l = on_mat_l

    def fit_simple(self, size=5, dxy=2.0):
        if self.dat is None:
            raise ValueError
        if self.init_center_r is None or self.init_center_l is None:
            raise ValueError
        if self.dat_gaussian_1 is None:
            self.gaussian_filter()

        nf = self.shape[self.ind_f]

        start_time = time.time()
        shape = self.shape
        on_dilated = np.zeros(shape, dtype=np.int8)
        dat_flt = self.dat_gaussian_1
        dat_flt_fitted = np.zeros(shape, dtype=dat_flt.dtype)

        for ys_lr, on_mat_lr in (
                (self.init_center_r, self.on_mat_r),
                (self.init_center_l, self.on_mat_l)
                ):
            for f, y in ( (tmp_f, tmp_y) for tmp_f in range(nf) for tmp_y in range(ys_lr[1]-ys_lr[0]+1) ):
                #point = on_mat_lr[f, y-ys_lr[0], :3]
                point = on_mat_lr[f, y, :3].astype(int)

                # 2D slice around a centerpoint
                xs = [max([0,point[self.ind_x]-size]), min([shape[self.ind_x],point[self.ind_x]+size+1])]
                ys = point[self.ind_y]
                zs = [max([0,point[self.ind_z]-size]), min([shape[self.ind_z],point[self.ind_z]+size+1])]

                xn = xs[1]-xs[0]-1
                yn = zs[1]-zs[0]-1
                
                dat_on = dat_flt[xs[0]:xs[1], ys, zs[0]:zs[1], f].copy()
                box_size = (2,2)
                search_x = max([box_size[0]/2, size-dxy]), min([xs[1]-1-box_size[0]/2, size+dxy])
                search_z = max([box_size[1]/2, size-dxy]), min([zs[1]-1-box_size[1]/2, size+dxy])

                #search_domain = [ (xx, yy) for xx in np.arange((box_size[0]-1), dat_on.shape[0]-(box_size[0]-1)+0.01, 0.2)
                #                           for yy in np.arange((box_size[1]-1), dat_on.shape[1]-(box_size[1]-1)+0.01, 0.2) ]
                search_domain = [ (xx, yy) for xx in np.arange((search_x[0]), search_x[1]+0.01, 0.2)
                                           for yy in np.arange((search_z[0]), search_z[1]+0.01, 0.2) ]

                
                f_interp = scipy.interpolate.interp2d(range(dat_on.shape[1]), range(dat_on.shape[0]),
                                                      dat_on, kind='cubic', fill_value=0.0)


                # use optimization?
                max_sum = 0.0
                max_ind = -1
                for i, (xx, yy) in enumerate(search_domain):
                    sum_i = f_interp(
                                     np.arange(yy-(box_size[1]-1), yy+(box_size[1]-1)+0.1, 1.0),
                                     np.arange(xx-(box_size[0]-1), xx+(box_size[0]-1)+0.1, 1.0)
                    )
                    if sum_i.sum() > max_sum:
                        max_sum = sum_i.sum()
                        max_ind = i
                imax = search_domain[max_ind]
                
                center_x, center_y = imax

                on_mat_lr[f, ys-ys_lr[0], 3:12] = 0
                # x, z swap in popt
                on_mat_lr[f, ys-ys_lr[0], 5] += (zs[0] + center_y)
                on_mat_lr[f, ys-ys_lr[0], 6] += (xs[0] + center_x)
                #on_mat_lr[f, ys-ys_lr[0], 12:] = [len(z.nonzero()[0]), rss_mean]
                #on_mat_lr[f, ys-ys_lr[0], 12:] = [len(z.nonzero()[0]), 0]
                on_mat_lr[f, ys-ys_lr[0], 12:] = [9, 0]

        print '--- %s seconds ---' % (time.time() - start_time)
        # x0, y0, z0
        # amp1, amp2, xo, yo, sigma_x1, sigma_x2, sigma_y1, sigma_y2, theta
        # num(seg), rss_mean
        #self.dat_edge_fitted = dat_edge_fitted



    def fit(self, size=5, thr=0.4, dxy=2.0):
        '''
size=4,
thr=0.4,
'''
        if self.dat is None:
            raise ValueError
        if self.init_center_r is None or self.init_center_l is None:
            raise ValueError
        if self.dat_edge is None:
            self.edge_filter()

        bounds = (
                (0.01, 5),
                (0.01, 5),
                (-dxy, dxy),
                (-dxy, dxy),
                (0.01, None),
                (0.01, None),
                (0.01, None),
                (0.01, None),
                (None, None)
        )
        filter_box = create_filter_box(size=size)
        filter_box_2 = create_filter_box(size=size, radius_o=3, radius_i=2)

        nf = self.shape[self.ind_f]

        start_time = time.time()
        shape = self.shape
        on_dilated = np.zeros(shape, dtype=np.int8)
        dat_edge = self.dat_edge
        dat_edge_fitted = np.zeros(shape, dtype=dat_edge.dtype)

        for ys_lr, on_mat_lr in (
                (self.init_center_r, self.on_mat_r),
                (self.init_center_l, self.on_mat_l)
                ):
            for f, y in ( (tmp_f, tmp_y) for tmp_f in range(nf) for tmp_y in range(ys_lr[1]-ys_lr[0]+1) ):
                #point = on_mat_lr[f, y-ys_lr[0], :3]
                point = on_mat_lr[f, y, :3].astype(int)

                # 2D slice around a centerpoint
                xs = [max([0,point[self.ind_x]-size]), min([shape[self.ind_x],point[self.ind_x]+size+1])]
                ys = point[self.ind_y]
                zs = [max([0,point[self.ind_z]-size]), min([shape[self.ind_z],point[self.ind_z]+size+1])]

                xn = xs[1]-xs[0]-1
                yn = zs[1]-zs[0]-1
                
                on_edge = dat_edge[xs[0]:xs[1], ys, zs[0]:zs[1], f].copy()
                
                on_edge *= filter_box[max([0,-point[self.ind_x]+size]):min([filter_box.shape[0],shape[self.ind_x]-point[self.ind_x]+size]),
                                      max([0,-point[self.ind_z]+size]):min([filter_box.shape[1],shape[self.ind_z]-point[self.ind_z]+size])]
                on_edge /= on_edge.max()

                # Create x and y indices
                xx = np.linspace(0, xn, xn+1)
                yy = np.linspace(0, yn, yn+1)
                xx, yy = np.meshgrid(yy, xx)
                
                # fit
                # amp1, amp2, xo, yo, sigma_x1, sigma_x2, sigma_y1, sigma_y2, theta
                #initial_guess = (2, 0, xn/2, yn/2, 1, 2, 1, 2, 0)
                initial_guess = (2.0, 2.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0)
                #if True:
                try:
                    res = opt.minimize(
                            lambda x:to_min_two_2d_Gaussian_base(x, (xx, yy), (xn, yn), on_edge),
                            initial_guess,
                            method='SLSQP',
                            jac=False,
                            options={'maxiter':1000},
                            bounds=bounds)
                            #constraints=constraints)
                            
                    if res.success is False:
                        raise ValueError

                    popt = res.x
                    popt[2:4] += [yn/2, xn/2]
                    
                    #popt, pcov = opt.curve_fit(two_2d_Gaussian, (xx, yy), on_edge.ravel(), p0=initial_guess, maxfev=10000)
                    on_edge_fitted = two_2d_Gaussian((xx, yy), *popt).reshape(xn+1, yn+1)
                    #rss_mean = ((on_edge - on_edge_fitted)**2).sum()/(on_edge.shape[0]*on_edge.shape[1])
                    #if rss_mean > 0.02:
                    #    raise
                
                #if False:
                except:
                    try:
                        on_edge = dat_edge[xs[0]:xs[1], ys, zs[0]:zs[1], f].copy()

                        on_edge *= filter_box_2[max([0,-point[0]+size]):min([filter_box.shape[0],shape[0]-point[0]+size]),
                                              max([0,-point[2]+size]):min([filter_box.shape[1],shape[2]-point[2]+size])]
                        on_edge /= on_edge.max()
                        
                        res = opt.minimize(
                                lambda x:to_min_two_2d_Gaussian_base(x, (xx, yy), (xn, yn), on_edge),
                                initial_guess,
                                method='SLSQP',
                                jac=False,
                                options={'maxiter':1000},
                                bounds=bounds)
                                #constraints=constraints)
                                
                        if res.success is False:
                            raise ValueError

                        popt = res.x
                        popt[2:4] += [yn/2, xn/2]
                    
                        #popt, pcov = opt.curve_fit(two_2d_Gaussian, (xx, yy), on_edge.ravel(), p0=initial_guess, maxfev=10000)
                        on_edge_fitted = two_2d_Gaussian((xx, yy), *popt).reshape(xn+1, yn+1)
                        #rss_mean = ((on_edge - on_edge_fitted)**2).sum()/(on_edge.shape[0]*on_edge.shape[1])
                        #if rss_mean > 0.02:
                        #    raise
                    except:
                        on_mat_lr[f, ys-ys_lr[0],:] = self.value_out
                        continue
                   
                amp1, amp2, xo, yo, sigma_x1, sigma_x2, sigma_y1, sigma_y2, theta = popt
                #on_edge_fitted = two_2d_Gaussian((xx, yy), *popt).reshape(xn+1, yn+1)
                dat_edge_fitted[xs[0]:xs[1], ys, zs[0]:zs[1], f] = on_edge_fitted
                
                
                # fill ON
                z = (on_edge_fitted > thr).astype(np.int8)
                for ind_x in range(z.shape[0]):
                    y_x = z[ind_x,:].nonzero()[0]
                    if len(y_x) == 0:
                        continue
                    for ind_y in range(min(y_x), max(y_x)+1):
                        z[ind_x, ind_y] = 1

                for ind_y in range(z.shape[1]):
                    x_y = z[:,ind_y].nonzero()[0]
                    if len(x_y) == 0:
                        continue
                    for ind_x in range(min(x_y), max(x_y)+1):
                        z[ind_x, ind_y] = 1

                on_dilated[xs[0]:xs[1], ys, zs[0]:zs[1], f] = z
                
                on_mat_lr[f, ys-ys_lr[0], 3:12] = popt
                # x, z swap in popt
                on_mat_lr[f, ys-ys_lr[0], 5] += zs[0]
                on_mat_lr[f, ys-ys_lr[0], 6] += xs[0]
                #on_mat_lr[f, ys-ys_lr[0], 12:] = [len(z.nonzero()[0]), rss_mean]
                on_mat_lr[f, ys-ys_lr[0], 12:] = [len(z.nonzero()[0]), 0]

        print '--- %s seconds ---' % (time.time() - start_time)
        # x0, y0, z0
        # amp1, amp2, xo, yo, sigma_x1, sigma_x2, sigma_y1, sigma_y2, theta
        # num(seg), rss_mean
        self.dat_edge_fitted = dat_edge_fitted

    def outlier(self):
        self.outlier_r = np.empty( (self.shape[self.ind_f], self.init_center_r[1] - self.init_center_r[0] + 1), dtype=bool )
        self.outlier_l = np.empty( (self.shape[self.ind_f], self.init_center_l[1] - self.init_center_l[0] + 1), dtype=bool )

        for ys_lr, on_mat_lr_all, outlier in (
                (self.init_center_r, self.on_mat_r, self.outlier_r),
                (self.init_center_l, self.on_mat_l, self.outlier_l)
                ):

            #on_mat_lr = on_mat_lr_all[:,:,3:]
            on_mat_lr = on_mat_lr_all[:,:,3:-1]

            q1 = np.percentile(on_mat_lr, 25, 0)
            q3 = np.percentile(on_mat_lr, 75, 0)
            iq = q3 - q1
            inner_fence = (q1 - 1.5*iq, q3 + 1.5*iq)
            outer_fence = (q1 - 3*iq, q3 + 3*iq)

            outlier_tmp = ((on_mat_lr == self.value_out) + (on_mat_lr > outer_fence[1]) + (on_mat_lr < outer_fence[0])).sum(2) > 0
            #outlier_tmp = ((on_mat_lr == self.value_out) + (on_mat_lr > inner_fence[1]) + (on_mat_lr < inner_fence[0])).sum(2) > 0

            outlier[:] = outlier_tmp

        print 'outliers: ', len(self.outlier_r.nonzero()[0]), len(self.outlier_l.nonzero()[0])
        print '  out of: ', len(self.outlier_r.nonzero()[0]) + len((self.outlier_r==0).nonzero()[0]), len(self.outlier_l.nonzero()[0]) + len((self.outlier_l==0).nonzero()[0])

    def estimate_from_nbd(self, on_mat_lr, f, y, outlier_update, thr=4):
        est = np.zeros(2)
        n_est = 0
        ny = on_mat_lr.shape[1]
        nf = self.shape[self.ind_f]

        if 0 < f:
            if outlier_update[f-1, y] == False:
            #if on_mat_lr[f-1, y, 5] >= 0:
            #if on_mat_lr[f-1, y, 5] != self.value_out:
                est += on_mat_lr[f-1, y, 5:7]
                n_est += 1
        if f < nf-1:
            if outlier_update[f+1, y] == False:
            #if on_mat_lr[f+1, y, 5] >= 0:
            #if on_mat_lr[f+1, y, 5] != self.value_out:
                est += on_mat_lr[f+1, y, 5:7]
                n_est += 1
        if 0 < y:
            if outlier_update[f, y-1] == False:
            #if on_mat_lr[f, y-1, 5] >= 0:
            #if on_mat_lr[f, y-1, 5] != self.value_out:
                est += on_mat_lr[f, y-1, 5:7]
                n_est += 1
        if  y < ny-1:
            if outlier_update[f, y+1] == False:
            #if on_mat_lr[f, y+1, 5] >= 0:
            #if on_mat_lr[f, y+1, 5] != self.value_out:
                est += on_mat_lr[f, y+1, 5:7]
                n_est += 1
        if n_est >= thr:
            return est / n_est
        return (-1, -1)

    def estimate(self):
        ny = self.shape[self.ind_y]
        nf = self.shape[self.ind_f]

        #for ys_lr, on_mat_lr_all, outlier in (
        for ys_lr, on_mat_lr, outlier in (
                (self.init_center_r, self.on_mat_r, self.outlier_r),
                (self.init_center_l, self.on_mat_l, self.outlier_l)
                ):
            #on_mat_lr = on_mat_lr_all[:,:,3:]
            #dat_mean = on_mat_lr[outlier == False].mean(0)

            outlier_voxels = zip(*outlier.nonzero())
            outlier_update = outlier.copy()

            updated = []
            while True:
                to_update = {}
                for thr in [4, 3, 2, 1]:
                    for (f, y) in outlier_voxels:
                        if (f, y) in updated:
                            continue
                        if to_update.has_key((f, y)):
                            continue
                        # use adj.slices, frames
                        est = self.estimate_from_nbd(on_mat_lr, f, y, outlier_update, thr=thr)
                        if est[0] != -1:
                            to_update[(f, y)] = est

                if len(to_update) == 0:
                    break

                for (f, y) in to_update.keys():
                    on_mat_lr[f, y, 5] = to_update[(f, y)][0]
                    on_mat_lr[f, y, 6] = to_update[(f, y)][1]
                    if to_update[(f, y)][0] < 0 or to_update[(f, y)][1] < 0:
                        print (f, y), to_update[(f, y)]
                    updated.append((f, y))
                    outlier_update[f, y] = False

            #on_mat_lr_all[:,:,3:] = on_mat_lr

            if len(updated) != len(outlier_voxels):
                print 'not all outliers were estimated'
                print 'estimate: ', len(updated)
                print 'outlier: ', len(outlier_voxels)

                #raise ValueError
            else:
                print len(updated)
    
    def make_center_0(self, shape_new=None):
        shape = self.shape
        if shape_new is None:
            shape_new = shape[:3]
            factor = [1.0, 1.0, 1.0]
        else:
            factor = [float(shape_new[i]) / shape[i] for i in range(len(shape_new))]

        on_dilated = np.zeros(shape_new, dtype=np.int8)

        for color, (ys_lr, on_mat_lr) in enumerate([
                (self.init_center_r, self.on_mat_r),
                (self.init_center_l, self.on_mat_l)]
                ):

            for y in range(ys_lr[1]-ys_lr[0]):
                y_new_0 = int(np.round((y + ys_lr[0]) * factor[1]))
                y_new_1 = int(np.round((y + 1 + ys_lr[0]) * factor[1]))
                x_new_0 = on_mat_lr[0, y, 6] * factor[0]
                x_new_1 = on_mat_lr[0, y + 1, 6] * factor[0]
                dx_new = x_new_1 - x_new_0
                z_new_0 = on_mat_lr[0, y, 5] * factor[2]
                z_new_1 = on_mat_lr[0, y + 1, 5] * factor[2]
                dz_new = z_new_1 - z_new_0
                n_y_new = float(y_new_1 - y_new_0) + 1

                for ind_y, y_new in enumerate(range(y_new_0, y_new_1)):
                    point = (
                            int(np.round(x_new_0 + ind_y/n_y_new*dx_new)),
                            int(np.round(y_new)),
                            int(np.round(z_new_0 + ind_y/n_y_new*dz_new))
                    )
                    on_dilated[point] = color+1

            # add last point
            if True:
                for y in [ys_lr[1]-ys_lr[0]]:
                    point = (
                            int(np.round(on_mat_lr[0, y, 6] * factor[0])),
                            int(np.round((y + ys_lr[0]) * factor[1])),
                            int(np.round(on_mat_lr[0, y, 5] * factor[2]))
                    )
                    on_dilated[point] = color+1

            # old version
            if False:
                for y in range(ys_lr[1]-ys_lr[0]+1):
                    point = (
                            int(np.round(on_mat_lr[0, y, 6] * factor[0])),
                            int(np.round((y + ys_lr[0]) * factor[1])),
                            int(np.round(on_mat_lr[0, y, 5] * factor[2]))
                    )
                    on_dilated[point] = color+1

        return on_dilated
   

    def make_center_frames(self, size=5):
        shape = self.shape

        dat_center_frames_l = np.zeros((size, self.init_center_l[1]-self.init_center_l[0]+1, size, shape[3]), dtype=np.float)
        dat_center_frames_r = np.zeros((size, self.init_center_r[1]-self.init_center_r[0]+1, size, shape[3]), dtype=np.float)

        for color, (ys_lr, on_mat_lr, dat_lr) in enumerate([
                (self.init_center_r, self.on_mat_r, dat_center_frames_r),
                (self.init_center_l, self.on_mat_l, dat_center_frames_l)]
                ):
            for y in range(ys_lr[1]-ys_lr[0]+1):
                for f in range(shape[3]):
                    y_whole = y + ys_lr[0]
                    f_interp = scipy.interpolate.interp2d(range(shape[2]), range(shape[0]),
                                                          self.dat[:,y_whole,:,f], kind='cubic', fill_value=0.0)
                    x_whole = on_mat_lr[f, y, 6]
                    z_whole = on_mat_lr[f, y, 5]
                    dat_lr[:, y, :, f] = f_interp(
                            np.arange(z_whole-size/2, z_whole+size/2+0.1, 1),
                            np.arange(x_whole-size/2, x_whole+size/2+0.1, 1)
                    )

        return dat_center_frames_r, dat_center_frames_l
    


    #def make_segmentation(self, size=5, thr=0.7):
    def make_segmentation(self, size=5, thr=0.7, gaussian_output=False, thr_low=0.1):
        '''
thr=0.6,
'''
        nf = self.shape[self.ind_f]

        shape = self.shape
        #on_dilated = np.zeros(shape, dtype=np.int8)
        if gaussian_output:
            on_dilated = np.zeros(shape, dtype=np.float32)
        else:
            on_dilated = np.zeros(shape, dtype=np.int8)
        dat_edge = self.dat_edge

        for ys_lr, on_mat_lr, outlier in (
                (self.init_center_r, self.on_mat_r, self.outlier_r),
                (self.init_center_l, self.on_mat_l, self.outlier_l)
                ):
            for y in range(ys_lr[1]-ys_lr[0]+1):
                mat_y = on_mat_lr[:, y, :]
                #out_y = outlier[:, y]
                
                #popt = mat_y[out_y == False].mean(0)[3:12]
                #popt = np.median(mat_y[out_y == False], 0)[3:12]
                #popt = np.mean(mat_y[out_y == False], 0)[3:12]
                popt = np.array([2, 2, 5, 5, 1, 2, 1, 2, 0], dtype=np.float32)

                #popt = on_mat_lr[:, y, 3:12].mean(0)
                
                for f in range(nf):
                    #point = on_mat_lr[f, y-ys_lr[0], :3]
                    #point = on_mat_lr[f, y, :3]
                    # x, z swap in popt
                    point = np.array([on_mat_lr[f, y, 6], y, on_mat_lr[f, y, 5]]).astype(int)

                    # 2D slice around a centerpoint
                    xs = [max([0,point[self.ind_x]-size]), min([shape[self.ind_x],point[self.ind_x]+size+1])]
                    ys = point[self.ind_y] + ys_lr[0]
                    zs = [max([0,point[self.ind_z]-size]), min([shape[self.ind_z],point[self.ind_z]+size+1])]

                    xn = xs[1]-xs[0]-1
                    yn = zs[1]-zs[0]-1
                    
                    #on_edge = dat_edge[xs[0]:xs[1], ys, zs[0]:zs[1], f].copy()
                    
                    # Create x and y indices
                    xx = np.linspace(0, xn, xn+1)
                    yy = np.linspace(0, yn, yn+1)
                    xx, yy = np.meshgrid(yy, xx)
                    
                    #amp1, amp2, xo, yo, sigma_x1, sigma_x2, sigma_y1, sigma_y2, theta = popt
                    #popt = on_mat_lr[f, ys_ys_lr[0], 3:12]
                    # x, z swap in popt
                    popt[2] = on_mat_lr[f, y, 5] - zs[0]
                    popt[3] = on_mat_lr[f, y, 6] - xs[0]
                    
                    if gaussian_output:
                        on_edge_fitted = twoD_Gaussian_raw((xx, yy), popt[0], popt[2], popt[3], popt[5], popt[7], popt[8]).ravel().reshape(xn+1, yn+1)
                        on_edge_fitted[on_edge_fitted<thr_low] = 0.0
                        on_dilated[xs[0]:xs[1], ys, zs[0]:zs[1], f] = on_edge_fitted

                    else:
                        on_edge_fitted = two_2d_Gaussian((xx, yy), *popt).reshape(xn+1, yn+1)
                    
                        # fill ON
                        z = (on_edge_fitted > thr).astype(np.int8)
                        for ind_x in range(z.shape[0]):
                            y_x = z[ind_x,:].nonzero()[0]
                            if len(y_x) == 0:
                                continue
                            for ind_y in range(min(y_x), max(y_x)+1):
                                z[ind_x, ind_y] = 1

                        for ind_y in range(z.shape[1]):
                            x_y = z[:,ind_y].nonzero()[0]
                            if len(x_y) == 0:
                                continue
                            for ind_x in range(min(x_y), max(x_y)+1):
                                z[ind_x, ind_y] = 1

                        on_dilated[xs[0]:xs[1], ys, zs[0]:zs[1], f] = z

                    #if f == 0 and y == 11:
                    #    print popt[2:4]
                    #    print xs, zs
                    #    #return on_dilated
                    #    return on_edge_fitted, self.dat_edge[xs[0]:xs[1], ys, zs[0]:zs[1], f]


        # x0, y0, z0
        # amp1, amp2, xo, yo, sigma_x1, sigma_x2, sigma_y1, sigma_y2, theta
        # num(seg), rss_mean
        return on_dilated
    

    def save(self, filename):
        with open(filename, 'wb') as f:
            #for var in self.__store__:
            #    pickle.dump(var, f)
            #    #np.save(f, var)

            pickle.dump(self.dat           , f)
            pickle.dump(self.aff           , f)
            pickle.dump(self.hdr           , f)
            pickle.dump(self.init_center_r , f)
            pickle.dump(self.init_center_l , f)
            pickle.dump(self.on_mat_r      , f)
            pickle.dump(self.on_mat_l      , f)
            pickle.dump(self.outlier_r     , f)
            pickle.dump(self.outlier_l     , f)
            pickle.dump(self.dat_edge      , f)
            pickle.dump(self.centers       , f)
            pickle.dump(self.coeffs        , f)
            pickle.dump(self.num_seg       , f)
            pickle.dump(self.rss_mean      , f)
            pickle.dump(self.shape         , f)


    def load(self, filename):
        with open(filename) as f:
            #for var in self.__store__:
            #    var = pickle.load(f)
            #    #var = np.load(f)

            self.dat = pickle.load(f)
            self.aff = pickle.load(f)
            self.hdr = pickle.load(f)
            self.init_center_r = pickle.load(f)
            self.init_center_l = pickle.load(f)
            self.on_mat_r = pickle.load(f)
            self.on_mat_l = pickle.load(f)
            self.outlier_r = pickle.load(f)
            self.outlier_l = pickle.load(f)
            self.dat_edge = pickle.load(f)
            self.centers = pickle.load(f)
            self.coeffs = pickle.load(f)
            self.num_seg = pickle.load(f)
            self.rss_mean = pickle.load(f)
            self.shape = pickle.load(f)



if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write('Usage: %s filename_dwi [filename_centerline=filename_on_centerline.nii.gz]\n' % sys.argv[0])
        sys.exit(-1)

    filename = sys.argv[1]
    if len(sys.argv) > 2:
        filename_on_centerline = sys.argv[2]
    else:
        filename_on_centerline = on_dbsi_utils.filename_wo_ext(filename) + '_on_centerline.nii.gz'

    filename_edge = on_dbsi_utils.filename_wo_ext(filename) + '_edge.nii.gz'
    filename_on_dilated = on_dbsi_utils.filename_wo_ext(filename) + '_on.nii.gz'
    filename_on_edge_fitted = on_dbsi_utils.filename_wo_ext(filename) + '_on_edge_fitted.nii.gz'

    img = nib.load(filename)
    run(img.get_data(), nib.load(filename_on_centerline).get_data(), img.get_affine(), img.get_header(), filename_edge, filename_on_dilated, filename_on_edge_fitted)

def run(dat, dat_cnt, aff, hdr, filename_edge, filename_on_dilated, filename_on_edge_fitted):
    dat_flt, dat_edge = filter_edge(dat)
    on_dbsi_utils.save_img(dat_edge, aff, hdr, filename_edge)

    dat_on_dilated, dat_edge_fitted, on_slice_info, on_y_l, on_y_r = fit_on(dat_cnt, dat_edge, thr=0.5, filter_box=create_filter_box())

    on_dbsi_utils.save_img(dat_on_dilated, aff, hdr, filename_on_dilated)
    on_dbsi_utils.save_img(dat_edge_fitted, aff, hdr, filename_on_edge_fitted)

def stats(on_slice_info, on_y_l, on_y_r, shape):
    # on_slice_info: point[0], point[2], popt, n_nz, rss_mean
    # popt: amp1, amp2, xo, yo, sigma_x1, sigma_x2, sigma_y1, sigma_y2, theta

    # stats: xo, yo, n_nz, rss_mean
    N = 9 + 2
    n_slice, n_frame = shape[1], shape[3]

    data = np.zeros( (2, n_slice, n_frame, N), dtype=float)

    for lr, lr_char, on_y in ((0, 'l', on_y_l), (1, 'r', on_y_r)):
        for y in range(n_slice):
            for ind in range(len(on_y[y])):
                f = on_y[y][ind][0]
                key = (f, y, lr_char)
                _, _, popt, n_nz, rss_mean = on_slice_info[key]
                data[lr, y, f, :] = [n_nz, rss_mean] + popt

    # outlier: boxplot, mean +- 3std
    outlier = np.zeros(data.shape[:-1], dtype=bool)
    for lr in (0,1):
        for y in range(n_slice):
            for i in (0, 1, 4, 5):
                data_sub = data[lr, y, :, i]
                outlier[lr, y, :] = np.logical_or(outlier[lr, y, :], calculate_outlier(data_sub))

    #mean_popt = data.mean(2)
    mean_popt = np.zeros((2, n_slice, N), dtype=float)
    for lr in (0,1):
        for y in range(n_slice):
            mean_popt[lr, y, :] = data[lr, y, :, :][outlier[lr, y, :]].mean()

    # normalize: sigmas
    for lr in (0,1):
        for y in range(n_slice):
            for f in range(n_frame):
                if outlier[lr, y, f] is True:
                    print 'estimating %s, slice %s, frame %s' % (lr, y, f)
                    est = estimate_xy(data, outlier, lr, y, f)
                    if est is not None:
                        x, y = est
                    else:
                        # estimation failed
                        continue
                # use mean coefficients

def calculate_outlier(data_sub):
# TODO
    pass

def interpolate(x0, x1, d0, d1):
    '''
    d0 = x-x0
    d1 = x-x1
    '''
    return (x1*d0 - x0*d1) / (d0-d1)

def search_neighbor(is_in, x, xmax, xmin=0):
    x_l = x - 1
    while x_l >= 0:
        if is_in(x_l):
            break
        x_l -= 1
    if x_l >= 0:
        x_r = x+1
    else:
        x_r = x_l+1
    while x_r <= xmax:
        if is_in(x_r):
            break
        x_r += 1
    if x_r > xmax:
        return None, None
    return x_l, x_r

def estimate_xy(data, outlier, lr, y, f):
    indX = 4
    indY = 5
    y_l, y_r = search_neighbor(lambda x:not outlier[lr,x,f], y, data.shape[1])
    f_l, f_r = search_neighbor(lambda x:not outlier[lr,l,x], y, data.shape[2])
    estimate_X = 0.0
    estimate_Y = 0.0
    weight_X = 0.0
    weight_Y = 0.0

    if y_l is not None:
        fit_y_X = interpolate(data[lr, y_l, :, indX], data[lr, y_r, :, indX], y-y_l, y-y_r)
        fit_y_Y = interpolate(data[lr, y_l, :, indY], data[lr, y_r, :, indY], y-y_l, y-y_r)
        weight_y_X = 1.0/(fit_y_X - data[lr, y, :, indX]).std()
        weight_y_Y = 1.0/(fit_y_Y - data[lr, y, :, indY]).std()
        estimate_X += weight_y_X * interpolate(data[lr, y_l, f, indX], data[lr, y_r, f, indX], y-y_l, y-y_r)
        estimate_Y += weight_y_Y * interpolate(data[lr, y_l, f, indY], data[lr, y_r, f, indY], y-y_l, y-y_r)
        weight_X += weight_y_X
        weight_Y += weight_y_Y

    if f_l is not None:
        fit_f_X = interpolate(data[lr, :, f_l, indX], data[lr, :, f_r, indX], f-f_l, f-f_r)
        fit_f_Y = interpolate(data[lr, :, f_l, indY], data[lr, :, f_r, indY], f-f_l, f-f_r)
        weight_f_X = 1.0/(fit_f_X - data[lr, :, f, indX]).std()
        weight_f_Y = 1.0/(fit_f_Y - data[lr, :, f, indY]).std()
        estimate_X += weight_y_X * interpolate(data[lr, y_l, f, indX], data[lr, y_r, f, indX], y-y_l, y-y_r)
        estimate_Y += weight_y_Y * interpolate(data[lr, y_l, f, indY], data[lr, y_r, f, indY], y-y_l, y-y_r)
        weight_X += weight_f_X
        weight_Y += weight_f_Y

    if weight == 0.0:
        print 'estimating xy failed.'
        return False

    x = estimating_X / weight_X
    y = estimating_Y / weight_Y
    
    return x, y

def filter_edge(dat):
    dat_flt = dat.copy()
    dat_edge = dat.copy()

    for frame in range(dat_edge.shape[-1]):
        dat_flt[:,:,:,frame] = ndimage.gaussian_filter(dat_flt[:,:,:,frame], sigma=0.5)

        # sobel on coronal slices
        #dx = ndimage.sobel(dat_edge[:,:,:,frame], 0)
        #dz = ndimage.sobel(dat_edge[:,:,:,frame], 2)
        dx = ndimage.sobel(dat_flt[:,:,:,frame], 0)
        dz = ndimage.sobel(dat_flt[:,:,:,frame], 2)
        #mag = np.hypot(dx, dy, dz)
        mag = np.hypot(dx, dz)
        
        # normalize
        #max_val = max([np.max(mag[p_1[0][0]-2:p_1[0][0]+3,p_1[0][1]-2:p_1[0][1]+3,p_1[0][2]-2:p_1[0][2]+3]),
        #               np.max(mag[p_2[0][0]-2:p_2[0][0]+3,p_2[0][1]-2:p_2[0][1]+3,p_2[0][2]-2:p_2[0][2]+3])])
        #max_val = max([np.max(mag[p_1[0][0]-2:p_1[0][0]+3,p_1[0][1],p_1[0][2]-2:p_1[0][2]+3]),
        #               np.max(mag[p_2[0][0]-2:p_2[0][0]+3,p_2[0][1],p_2[0][2]-2:p_2[0][2]+3])])
        max_val = (mag[p_1[0][0]-2:p_1[0][0]+3,p_1[0][1],p_1[0][2]-2:p_1[0][2]+3].mean() +
                mag[p_2[0][0]-2:p_2[0][0]+3,p_2[0][1],p_2[0][2]-2:p_2[0][2]+3].mean())/2.0
        mag *= 255.0 / max_val
        dat_edge[:,:,:,frame] = mag

    return dat_flt, dat_edge

def create_filter_box(size=5, radius_o=4, radius_i=3):
    filter_box = np.ones((2*size+1,2*size+1), dtype=float)
    for i in range(2*size+1):
        for j in range(2*size+1):
            if (size-i)**2 + (size-j)**2 > radius_o**2:
                filter_box[i,j] = 0.4
            elif (size-i)**2 + (size-j)**2 > radius_i**2:
                filter_box[i,j] = 0.7
    return filter_box

def box(x0, y0, shape, size):
    xs = [max([0,x0-size]), min([x0,x0+size+1])]
    zs = [max([0,y0-size]), min([y0,y0+size+1])]
    return xs, zs

def fit_on(dat_cnt, dat_edge, thr=0.5, filter_box=create_filter_box()):
    shape = dat_edge.shape
    dat_on_dilated = np.zeros(shape, dtype=np.int8)
    dat_edge_fitted = np.zeros(shape, dtype=dat_edge.dtype)

    # (f, y, 'lr') = (x, z, popt, length, rss)
    on_slice_info = {}
    # f, x, z
    on_y_l = [[] for tmp in range(shape[1])]
    on_y_r = [[] for tmp in range(shape[1])]

    for f in range(shape[3]):
        center_points = zip(*dat_cnt[:,:,:,f].nonzero())
        for point in center_points:
            if point[0] < shape[0]/2:
                on_y_r[ys].append( (f, point[0], point[2]) )
                key = (f, point[1], 'r')
            else:
                on_y_l[ys].append( (f, point[0], point[2]) )
                key = (f, point[1], 'l')

            # 2D slice around a centerpoint
            xs, zs = box(point[0], point[2], size, shape)
            ys = point[1]

            xn = xs[1]-xs[0]-1
            yn = zs[1]-zs[0]-1
            
            on_edge = dat_edge[xs[0]:xs[1], ys, zs[0]:zs[1], f].copy()
            
            on_edge *= filter_box[max([0,-point[0]+size]):min([filter_box.shape[0],shape[0]-point[0]+size]),
                                  max([0,-point[2]+size]):min([filter_box.shape[1],shape[2]-point[2]+size])]
            on_edge /= on_edge.max()

            # Create x and y indices
            x = np.linspace(0, xn, xn+1)
            y = np.linspace(0, yn, yn+1)
            x, y = np.meshgrid(y, x)
            
            # fit
            # amp1, amp2, xo, yo, sigma_x1, sigma_x2, sigma_y1, sigma_y2, theta
            initial_guess = (2, 0, xn/2, yn/2, 1, 2, 1, 2, 0)
            try:
                popt, pcov = opt.curve_fit(two_2d_Gaussian, (x, y), on_edge.ravel(), p0=initial_guess, maxfev=10000)
                on_edge_fitted = two_2d_Gaussian((x, y), *popt).reshape(xn+1, yn+1)
                rss_mean = ((on_edge - on_edge_fitted)**2).sum()/(on_edge.shape[0]*on_edge.shape[1])
                if rss_mean > 0.02:
                    raise
            except:
                try:
                    on_edge = dat_edge[xs[0]:xs[1], ys, zs[0]:zs[1], f].copy()

                    on_edge *= filter_box_2[max([0,-point[0]+size]):min([filter_box.shape[0],shape[0]-point[0]+size]),
                                          max([0,-point[2]+size]):min([filter_box.shape[1],shape[2]-point[2]+size])]
                    on_edge /= on_edge.max()
                    
                    popt, pcov = opt.curve_fit(two_2d_Gaussian, (x, y), on_edge.ravel(), p0=initial_guess, maxfev=10000)
                    on_edge_fitted = two_2d_Gaussian((x, y), *popt).reshape(xn+1, yn+1)
                    rss_mean = ((on_edge - on_edge_fitted)**2).sum()/(on_edge.shape[0]*on_edge.shape[1])
                    #if rss_mean > 0.02:
                    #    raise
                except:
                    on_slice_info[key] = (point[0], point[2], popt, -1, -1)
                    continue
               
            amp1, amp2, xo, yo, sigma_x1, sigma_x2, sigma_y1, sigma_y2, theta = popt
            on_edge_fitted = two_2d_Gaussian((x, y), *popt).reshape(xn+1, yn+1)
            dat_edge_fitted[xs[0]:xs[1], ys, zs[0]:zs[1], f] = on_edge_fitted
            
            
            # fill ON
            z = (on_edge_fitted > thr).astype(np.int8)
            for ind_x in range(z.shape[0]):
                y_x = z[ind_x,:].nonzero()[0]
                if len(y_x) == 0:
                    continue
                for ind_y in range(min(y_x), max(y_x)+1):
                    z[ind_x, ind_y] = 1

            for ind_y in range(z.shape[1]):
                x_y = z[:,ind_y].nonzero()[0]
                if len(x_y) == 0:
                    continue
                for ind_x in range(min(x_y), max(x_y)+1):
                    z[ind_x, ind_y] = 1

            dat_on_dilated[xs[0]:xs[1], ys, zs[0]:zs[1], f] = z
            
            on_slice_info[key] = (point[0], point[2], popt, len(z.nonzero()[0]), rss_mean)

    return dat_on_dilated, dat_edge_fitted, on_slice_info, on_y_l, on_y_r

#

