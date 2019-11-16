#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, argparse
from processing import append_mCSM

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name',type=str)
parser.add_argument('-C','--center',type=str, choices=['CA','geometric'],required=True)
args = parser.parse_args()
dataset_name = args.dataset_name
if args.center:
    center = args.center

mCNN_k_path = '/public/home/sry/mCNN/datasets_array/%s/k_neighbor'%dataset_name
mCNN_r_path = '/public/home/sry/mCNN/datasets_array/%s/radius'%dataset_name
mCSM_path   = '/public/home/sry/mCNN/datasets_array/%s/mCSM/%s'%(dataset_name,center)

mCNN_k_dirlst = [mCNN_k_path + '/' + x for x in os.listdir(mCNN_k_path) if os.path.isfile(x)]
mCNN_r_dirlst = [mCNN_r_path + '/' + x for x in os.listdir(mCNN_r_path) if os.path.isfile(x)]

