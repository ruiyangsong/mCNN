#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, argparse
parser = argparse.ArgumentParser()
parser.add_argument('qsub_path', type=str)
args = parser.parse_args()
path = args.qsub_path
flag = 0
dirlst = [path+'/'+x for x in os.listdir(path)]
for dir in dirlst:
    errdir = dir+'/qsublog/err'
    assert os.path.exists(errdir)
    if os.path.getsize(errdir) > 0:
        flag+=1
        print(errdir)
print('Number of errfile: %d'%flag)