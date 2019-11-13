#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', type=str)
parser.add_argument('center',       type=str, choices= ['CA', 'geometric'], default='geometric')
parser.add_argument('featureORpdb', type=str, choices=['feature','pdb'], default='feature')
args = parser.parse_args()
dataset_name = args.dataset_name
center = args.center
featureORpdb = args.featureORpdb

feature_path   = '/public/home/sry/mCNN/datasets/%s/csv_%s%s_%s' % (dataset_name, featureORpdb, dataset_name, center)
feature_dirlst = [feature_path+'/'+x for x in os.listdir(feature_path)]
for feature_dir in feature_dirlst:
    errdir = feature_dir+'/qsublog/err'
    assert os.path.exists(errdir)
    if os.path.getsize(errdir) > 0:
        print(errdir)