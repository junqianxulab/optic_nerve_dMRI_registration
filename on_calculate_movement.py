#!/usr/bin/env python

# coding: utf-8

import sys
import os

if len(sys.argv) < 2:
    sys.stderr.write('Usage: %s model_filename T|[F]\n' % os.path.basename(sys.argv[0]))
    sys.stderr.write('    T or F: whether printing mean displacement of each volume as well. T: print, F: do not print. default: F.\n')
    sys.exit(1)

fn_model = sys.argv[1]
if not os.path.isfile(fn_model):
    sys.stderr.write('%s not exist.\n' % fn_model)
    sys.exit(1)
if len(sys.argv) > 2:
    to_print_volume = sys.argv[2].lower() == 't'
else:
    to_print_volume = False

import on_model

on = on_model.OpticNerveFit()
on.load(fn_model)
mean_dist = [0,0]
mean_dist_volume = [0,0]
for i, on_mat_lr in enumerate([on.on_mat_l, on.on_mat_r]):
    mean_y = on_mat_lr[:,:,5].mean(0)
    mean_x = on_mat_lr[:,:,6].mean(0)
    distance = ((on_mat_lr[:,:,5]-mean_y)**2 + (on_mat_lr[:,:,6]-mean_x)**2)**0.5
    mean_dist[i] = distance.mean(0)
    mean_dist_volume[i] = distance.mean(1)

print "mean movement distance (voxel)"
print "slice(posterior-anterior),left/right,distance(voxel)"
for i, d in enumerate(mean_dist[0]):
    print "%s,left,%s" % (i, d)
for i, d in enumerate(mean_dist[1]):
    print "%s,right,%s" % (i, d)

if to_print_volume:
    print ""
    print "mean movement distance (voxel)"
    print "volume,left/right,distance(voxel)"
    for i, d in enumerate(mean_dist_volume[0]):
        print "%s,left,%s" % (i, d)
    for i, d in enumerate(mean_dist_volume[1]):
        print "%s,right,%s" % (i, d)


