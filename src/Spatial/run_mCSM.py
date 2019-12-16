#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, argparse, time

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name')
args = parser.parse_args()
dataset_name = args.dataset_name

app = '/public/home/sry/mCNN/src/Spatial/mCSM.py'
os.system('chmod 755 %s' % app)
wild_outpath = '/public/home/sry/mCNN/dataset/%s/feature/mCSM/wild'%dataset_name
mutant_outpath = '/public/home/sry/mCNN/dataset/%s/feature/mCSM/mutant'%dataset_name
if not os.path.exists(wild_outpath):
    os.makedirs(wild_outpath)
if not os.path.exists(mutant_outpath):
    os.makedirs(mutant_outpath)
class_numlst = [2,8]
centerlst = ['CA','geometric']
minimum = 0.1
maximunlst = [5,6,7,8,9,10,11,12,13,14,15]
# maximunlst = [14]
steplst    = [0.5, 1, 1.5, 2]

# for wild structure
wild_csv_feature_dir = '/public/home/sry/mCNN/dataset/S1925/feature/mCNN/wild/csv'
if not  os.path.exists(wild_csv_feature_dir):
    os.makedirs(wild_csv_feature_dir)
for class_num in class_numlst:
    for center in centerlst:
        for maximum in maximunlst:
            for step in steplst:
                qsubid = '%s_wild_min_%.1f_max_%.1f_step_%.1f_center_%s_class_%s'%(dataset_name,minimum,maximum,step,center,class_num)
                outdir = '%s/npz' % wild_outpath
                if not os.path.exists('%s/qsublog' % wild_outpath):
                    os.system('mkdir -p %s/qsublog' % wild_outpath)
                walltime = 'walltime=240:00:00'
                errfile = '%s/qsublog/err' % wild_outpath
                outfile = '%s/qsublog/out' % wild_outpath
                run_CalmCSM = '%s/run_CalmCSM.sh' % (outdir)

                g = open(run_CalmCSM, 'w+')
                g.writelines('#!/usr/bin/bash\n')
                g.writelines('dataset_name=\"%s\"\n' % dataset_name)
                g.writelines('echo $dataset_name\n')
                g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
                g.writelines('%s %s %s -o %s --min %s --max %s --step %s --center %s --class_num %s\n' % (app, dataset_name, wild_csv_feature_dir, outdir, minimum,maximum,step,center,class_num))
                g.writelines("echo 'end at:' `date`\n")
                g.close()
                os.system('chmod 755 %s' % run_CalmCSM)
                os.system('/public/home/sry/bin/getQ.pl')
                os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, qsubid, run_CalmCSM))
                time.sleep(0.1)

# for mutant_structure
mutant_csv_feature_dir = '/public/home/sry/mCNN/dataset/S1925/feature/mCNN/mutant/csv'
if not os.path.exists(mutant_csv_feature_dir):
    os.makedirs(mutant_csv_feature_dir)
for class_num in class_numlst:
    for center in centerlst:
        for maximum in maximunlst:
            for step in steplst:
                qsubid = '%s_mutant_min_%.1f_max_%.1f_step_%.1f_center_%s_class_%s'%(dataset_name,minimum,maximum,step,center,class_num)
                outdir = '%s/npz' % mutant_outpath
                if not os.path.exists('%s/qsublog' % mutant_outpath):
                    os.system('mkdir -p %s/qsublog' % mutant_outpath)
                walltime = 'walltime=240:00:00'
                errfile = '%s/qsublog/err' % mutant_outpath
                outfile = '%s/qsublog/out' % mutant_outpath
                run_CalmCSM = '%s/run_CalmCSM.sh' % (outdir)

                g = open(run_CalmCSM, 'w+')
                g.writelines('#!/usr/bin/bash\n')
                g.writelines('dataset_name=\"%s\"\n' % dataset_name)
                g.writelines('echo $dataset_name\n')
                g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
                g.writelines('%s %s %s -o %s --min %s --max %s --step %s --center %s --class_num %s\n' % (app, dataset_name, mutant_csv_feature_dir, outdir, minimum,maximum,step,center,class_num))
                g.writelines("echo 'end at:' `date`\n")
                g.close()
                os.system('chmod 755 %s' % run_CalmCSM)
                os.system('/public/home/sry/bin/getQ.pl')
                os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, qsubid, run_CalmCSM))
                time.sleep(0.1)