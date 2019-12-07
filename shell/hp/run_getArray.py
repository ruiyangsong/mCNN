#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name',type=str,help='dataset name')
parser.add_argument('-T','--PCA',type=str, default='False')
args = parser.parse_args()
dataset_name = args.dataset_name
if args.PCA:
    pca = args.PCA

app = '/public/home/sry/mCNN/src/getArray.py'
outdir_k = '/public/home/sry/mCNN/datasets_array/%s/k_neighbor'%dataset_name
outdir_r = '/public/home/sry/mCNN/datasets_array/%s/radius'%dataset_name

k_lst     = np.arange(30,351,10)
r_lst     = np.arange(3,21,1)
centerlst = ['CA', 'geometric']
# k_lst = [20]
# r_lst = [3]
# centerlst = ['CA']

for center in centerlst:
    for k_neighbor in k_lst:
        tag = '%s_center_%s_PCA_%s_neighbor_%s'%(dataset_name,center,pca,k_neighbor)
        qsubdir = '%s/qsub/%s' %(outdir_k,tag)
        if not os.path.exists(qsubdir):
            os.system('mkdir -p %s' %qsubdir)
        walltime = 'walltime=240:00:00'
        errfile = '%s/err'%qsubdir
        outfile = '%s/out'%qsubdir
        run_getArray = '%s/run_getArray.sh' % qsubdir
        g = open(run_getArray, 'w+')
        g.writelines('#!/usr/bin/bash\n')
        g.writelines('dataset_name=\"%s\"\n' % dataset_name)
        g.writelines('echo $dataset_name\n')
        g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
        g.writelines('%s %s -k %s -C %s -T %s\n' % (app, dataset_name,k_neighbor, center, pca))
        g.writelines("echo 'end at:' `date`\n")
        g.close()
        os.system('chmod 755 %s' % run_getArray)
        os.system('/public/home/sry/bin/getQ.pl')
        os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, tag, run_getArray))
        time.sleep(0.1)

    for radii in r_lst:
        tag = '%s_center_%s_PCA_%s_radius_%s' % (dataset_name, center, pca, radii)
        qsubdir = '%s/qsub/%s' % (outdir_r, tag)
        if not os.path.exists(qsubdir):
            os.system('mkdir -p %s' % qsubdir)
        walltime = 'walltime=240:00:00'
        errfile = '%s/err' % qsubdir
        outfile = '%s/out' % qsubdir
        run_getArray = '%s/run_getArray.sh' % qsubdir
        g = open(run_getArray, 'w+')
        g.writelines('#!/usr/bin/bash\n')
        g.writelines('dataset_name=\"%s\"\n' % dataset_name)
        g.writelines('echo $dataset_name\n')
        g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
        g.writelines('%s %s -r %s -C %s -T %s\n' % (app, dataset_name, radii, center, pca))
        g.writelines("echo 'end at:' `date`\n")
        g.close()
        os.system('chmod 755 %s' % run_getArray)
        os.system('/public/home/sry/bin/getQ.pl')
        os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, tag, run_getArray))
        time.sleep(0.1)