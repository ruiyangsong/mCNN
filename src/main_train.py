#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

dataset_name     = 'S2648'
val_dataset_name = 'S1925'
wild_or_mutant   = 'wild'# ['wild', 'mutant', 'stack']
center           = 'CA'
mCNN             = 'False 50'
# append           = 'False'
# mCSM             = '0.1 10 0.5 2'# min, max, step, atom_class_num
normalize        = 'norm'
sort             = 'chain'
random_seed      = '1 1 1'
model            = 'multi_task_conv2D'
Kfold            = 20
verbose          = 1
epoch            = 20
batch_size       = 128
CUDA             = '1'

print('Begin at: %s' %)
os.system('./Network/train.py %s %s %s -C %s --mCNN %s -n %s -s %s -d %s -D %s -K %s -V %s -E %s -B %s --CUDA %s'%(dataset_name, val_dataset_name, wild_or_mutant, center, mCNN, normalize, sort, random_seed, model, Kfold, verbose, epoch, batch_size, CUDA))
print('End at: %s' %)