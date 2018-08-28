import numpy as np
import math

def voxels_between_pts(point_0, point_1):
    if len(point_0) != 3 or len(point_1) != 3:
        return []

    dx = point_1[0] - point_0[0]
    dy = point_1[1] - point_0[1]
    dz = point_1[2] - point_0[2]

    bdry_x = np.arange( 0.5, np.abs(dx), 1.0 )
    bdry_y = np.arange( 0.5, np.abs(dy), 1.0 )
    bdry_z = np.arange( 0.5, np.abs(dz), 1.0 )

    bdry_x *= np.sign(dx)
    bdry_y *= np.sign(dy)
    bdry_z *= np.sign(dz)

    # P(t) = point_0 + t*(dx, dy, dz)
    t_x = [ tmp / dx for tmp in bdry_x ]
    t_y = [ tmp / dy for tmp in bdry_y ]
    t_z = [ tmp / dz for tmp in bdry_z ]

    # append a number to avoid empty t_i list
    t_x.append(2)
    t_y.append(2)
    t_z.append(2)

    t = 0
    voxels = [(point_0[0], point_0[1], point_0[2])]
    voxel = [point_0[0], point_0[1], point_0[2]]
    while t < 2:
        if t_x[0] < t_y[0]:
            if t_x[0] < t_z[0]:
                t = t_x.pop(0)
                voxel[0] += np.sign(dx)
            else:
                t = t_z.pop(0)
                voxel[2] += np.sign(dz)
        elif t_y[0] < t_z[0]:
            t = t_y.pop(0)
            voxel[1] += np.sign(dy)
        else:
            t = t_z.pop(0)
            voxel[2] += np.sign(dz)
        if t < 2:
            voxels.append(
                    (voxel[0],
                     voxel[1],
                     voxel[2]) )
        else:
            break

    if voxels[-1] != (point_1[0], point_1[1], point_1[2]):
        voxels.append((point_1[0], point_1[1], point_1[2]))
    return voxels

def nbds_6_wo_bdry_check(point):
    nbd_6 = ( (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1) )
    return [ (point[0]+nbd[0], point[1]+nbd[1], point[2]+nbd[2])
                for nbd in nbd_6 ]

def find_shared_nbd( point_0, point_1, shape=None ):
    nbd_0 = nbds_6_wo_bdry_check(point_0)
    nbd_1 = nbds_6_wo_bdry_check(point_1)
    shared_nbd = [nbd for nbd in nbd_0 if nbd in nbd_1]
    if shape is not None:
        shared_nbd = [nbd for nbd in shared_nbd
                    if nbd[0]>=0 and nbd[0] < shape[0]
                    and nbd[1]>=0 and nbd[1] < shape[1]
                    and nbd[2]>=0 and nbd[2] < shape[2]]

    return shared_nbd

def cross_section_area(points):
    cnt = points[0]
    ax, ay, az = cnt
    area = 0.0
    for i in range(1, len(points)-1):
        bx = points[i][0]
        by = points[i][1]
        bz = points[i][2]

        cx = points[i+1][0]
        cy = points[i+1][1]
        cz = points[i+1][2]

        tmp = 0.0
        det = ax*by - ax*cy + bx*cy - bx*ay + cx*ay - cx*by;
        tmp += det*det;
        det = ay*bz - ay*cz + by*cz - by*az + cy*az - cy*bz;
        tmp += det*det;
        det = az*bx - az*cx + bz*cx - bz*ax + cz*ax - cz*bx;
        tmp += det*det;

        area += ( np.sqrt( tmp ) / 2.0 );
    return area

def cube_cross_section(voxel, base, normal):
    vertices = np.array([ (voxel[0]+vert[0], voxel[1]+vert[1], voxel[2]+vert[2]) for vert in
                         [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5),
                          (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5)] ])
    edges = [ (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7) ]
    vectors = vertices - np.array(base)
    dots = np.dot(vectors, normal)
    cross_vertices = [ i for i in range(len(vertices)) if abs(dots[i])<0.001 ]
    candi_edges = [ edge for edge in edges if edge[0] not in cross_vertices and edge[1] not in cross_vertices ]

    cross_points = [ vertices[i] for i in cross_vertices ]
    for edge in candi_edges:
        if np.sign(dots[edge[0]]) == np.sign(dots[edge[1]]):
            continue
        v0 = vertices[edge[0]]
        v1 = vertices[edge[1]]
        u = np.dot(normal, (np.array(base)-v0)) / np.dot(normal, v1 - v0)
        if u > 1.0001 or u < -0.0001:
            print 'error ', voxel, base, normal, u
        vi = v0 + (v1-v0)*u
        #print v0, v1, vi

        cross_points.append(vi)

    cross_points = np.array(cross_points)
    if len(cross_points) < 3:
        return 0.0

    #print cross_points
    long_normal = np.argmax(normal)
    if long_normal == 0:
        cross_points_2d = cross_points[:,1:]
    elif long_normal == 1:
        cross_points_2d = cross_points[:,0:3:2]
    else:
        cross_points_2d = cross_points[:,:2]
    center = cross_points.mean(0)
    center_2d = cross_points_2d.mean(0)
    base_vector = cross_points_2d[0,:] - center_2d
    base_vector /= np.sqrt(np.dot(base_vector, base_vector))
    temp = np.array([ np.dot(cross_points_2d[i,:]-center_2d, base_vector) / np.sqrt(np.dot(cross_points_2d[i,:]-center_2d, cross_points_2d[i,:]-center_2d)) for i in range(len(cross_points_2d))])
    #print temp
    temp[temp>1] = 1.0
    temp[temp<-1] = -1.0
    angles = np.arccos( temp )
    signs = np.array([np.sign( (cross_points_2d[0][0] - center_2d[0]) * (a_point[1] - center_2d[1]) - (cross_points_2d[0][1] - center_2d[1]) * (a_point[0] - center_2d[0]) ) for a_point in cross_points_2d[1:,:]])
    angles[1:][signs<0] += np.pi
    #print angles*180/np.pi
    #print cross_points
    ordered = np.argsort(angles[1:])
    ordered_points = [ cross_points[order+1,:] for order in ordered ]
    ordered_points.insert(0, cross_points[0])
    ordered_points.append(cross_points[0])
    ordered_points.insert(0, center)
    #print ordered_points
    return cross_section_area(ordered_points)

def get_nbds_bdchk(voxel, nbds, nx, ny, nz):
    return [ (voxel[0]+nbd[0], voxel[1]+nbd[1], voxel[2]+nbd[2]) for nbd in nbds \
        if 0<=voxel[0]+nbd[0]<nx and 0<=voxel[1]+nbd[1]<ny and 0<=voxel[2]+nbd[2]<nz ]

def get_nbd_2d(voxel, nx, ny, nz, plane='y'):
    if plane == 1 or plane == 'y':
        nbds = [(1,0,0), (-1,0,0), (0,0,1), (0,0,-1)]
    elif plane == 0 or plane == 'x':
        nbds = [(0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    else:
        nbds = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0)]
    
    return [ (voxel[0]+nbd[0], voxel[1]+nbd[1], voxel[2]+nbd[2]) for nbd in nbds \
           if 0<=voxel[0]+nbd[0]<nx and 0<=voxel[1]+nbd[1]<ny and 0<=voxel[2]+nbd[2]<nz ]

def _center_of_mass(voxels, data=None):
    center_x = 0.0
    center_y = 0.0
    center_z = 0.0
    sum_ints = 0.0
    for voxel in voxels:
        if data is None:
            value = 1
        else:
            value = data[voxel]
        center_x += voxel[0] * value
        center_y += voxel[1] * value
        center_z += voxel[2] * value
        sum_ints += value
    return (center_x/sum_ints, center_y/sum_ints, center_z/sum_ints)

def center_of_mass(voxels, data=None, weight=lambda x:x):
    center_x = 0.0
    center_y = 0.0
    center_z = 0.0
    sum_ints = 0.0
    for voxel in voxels:
        if data is None:
            value = 1
        else:
            value = weight(data[voxel])
        center_x += voxel[0] * value
        center_y += voxel[1] * value
        center_z += voxel[2] * value
        sum_ints += value
    return (center_x/sum_ints, center_y/sum_ints, center_z/sum_ints)

def in_out(voxel, base, normal):
    vertices = np.array([ (voxel[0]+vert[0], voxel[1]+vert[1], voxel[2]+vert[2]) for vert in
                         [(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5),
                          (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)] ])
    vectors = vertices - np.array(base)
    dots = np.dot(vectors, normal)
    #return vertices, vectors, dots
    return (np.abs(dots)<0.001).any() or ( (dots>0).any() and (dots<0).any() )

def get_nbds(voxel, nbds):
    return [ (voxel[0]+nbd[0], voxel[1]+nbd[1], voxel[2]+nbd[2]) for nbd in nbds] 

def grad(voxel, data):
    dx = float(data[voxel[0]-1, voxel[1], voxel[2]] - data[voxel[0]+1, voxel[1], voxel[2]])/2
    dy = float(data[voxel[0], voxel[1]-1, voxel[2]] - data[voxel[0], voxel[1]+1, voxel[2]])/2
    dz = float(data[voxel[0], voxel[1], voxel[2]-1] - data[voxel[0], voxel[1], voxel[2]+1])/2
    sqrsum = np.sqrt(dx*dx + dy*dy + dz*dz)
    if sqrsum <= 0.0001:
        return (0.0,0.0,0.0)
    factor = 50
    dx /= factor
    dy /= factor
    dz /= factor
#    dx /= sqrsum
#    dy /= sqrsum
#    dz /= sqrsum
    return (dx, dy, dz)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2)
    b, c, d = -axis*math.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def dist(p0, p1):
    p0a = np.array(p0, dtype=float)
    p1a = np.array(p1, dtype=float)
    return np.sqrt(((p0a-p1a)**2).sum())


def dilate(base, nbd, num, nx, ny, nz, ymax=None, ymin=None):
    surface = []
    added = []
    for a_voxel in base:
        added.append(a_voxel)
    for i in range(num):
        todo = added[:]
        surface += added
        added = []

        for a_voxel in todo:
            for a_nbd in get_nbds_bdchk(a_voxel, nbd, nx, ny, nz):
                if a_nbd not in added and a_nbd not in surface:
                    if ymax is not None and a_nbd[1] > ymax:
                        continue
                    if ymin is not None and a_nbd[1] < ymin:
                        continue
                    added.append(a_nbd)
    surface += added
    return surface

def dilate_thr(base, nbd, num, nx, ny, nz, dat, thr, ymax=None, ymin=None):
    surface = []
    added = []
    for a_voxel in base:
        added.append(a_voxel)
    for i in range(num):
        todo = added[:]
        surface += added
        added = []

        for a_voxel in todo:
            for a_nbd in get_nbds_bdchk(a_voxel, nbd, nx, ny, nz):
                if a_nbd not in added and a_nbd not in surface:
                    if ymax is not None and a_nbd[1] > ymax:
                        continue
                    if ymin is not None and a_nbd[1] < ymin:
                        continue
                    if dat[a_nbd] > thr: 
                        added.append(a_nbd)
        if not added:
            break
    surface += added
    return surface

def dilate_2thr(base, nbd, num, nx, ny, nz, dat_g, thr_g, dat_l, thr_l, ymax=None, ymin=None):
    surface = []
    added = []
    for a_voxel in base:
        added.append(a_voxel)
    for i in range(num):
        todo = added[:]
        surface += added
        added = []

        for a_voxel in todo:
            for a_nbd in get_nbds_bdchk(a_voxel, nbd, nx, ny, nz):
                if a_nbd not in added and a_nbd not in surface:
                    if ymax is not None and a_nbd[1] > ymax:
                        continue
                    if ymin is not None and a_nbd[1] < ymin:
                        continue
                    if dat_g[a_nbd] > thr_g and dat_l[a_nbd] < thr_l: 
                        added.append(a_nbd)
        if not added:
            break
    surface += added
    return surface

def get_edge_pixel( point, cnt, crop_dx, crop_dy, shft ):
    pt = ( point[0] - cnt[0] + crop_dx, point[1] - cnt[1] + crop_dy )
    x0 = pt[0] - 0.5 + shft
    x1 = pt[0] + 0.5 + shft
    y0 = pt[1] - 0.5 + shft
    y1 = pt[1] + 0.5 + shft
    return (x0, y0, x0, y1), (x0, y0, x1, y0), (x0, y1, x1, y1), (x1, y0, x1, y1)


##############
# example nbds:
nbd_26 = [(ti,tj,tk) for ti in [-1,0,1] for tj in [-1,0,1] for tk in [-1,0,1]]
nbd_26.remove((0,0,0))

nbd_6 = [ (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1) ]

nbd_d1 = nbd_6
nbd_d3 = [ ijk for ijk in nbd_26 if ijk[0]*ijk[1]*ijk[2] != 0]
#nbd_d2 = [ ijk for ijk in nbd_26 if ijk not in nbd_d1 and if ijk[0]*ijk[1]*ijk[2] == 0]
nbd_d2 = [ ijk for ijk in nbd_26 if ijk not in nbd_d1 and ijk not in nbd_d3 ]
#

