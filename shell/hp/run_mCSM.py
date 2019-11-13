#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, argparse, time

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name')
args = parser.parse_args()
dataset_name = args.dataset_name

app = '/public/home/sry/mCNN/src/mCSM.py'
os.system('chmod 755 %s' % app)
outpath = '/public/home/sry/mCNN/datasets_array/%s/mCSM'%dataset_name

class_numlst = [2,8]
centerlst = ['CA','geometric']
minimum = 0.1
maximunlst = [5,6,7,8,9,10,11,12,13,14,15]
# maximunlst = [14]
steplst    = [0.5, 1, 1.5, 2]

for class_num in class_numlst:
    for center in centerlst:
        for maximum in maximunlst:
            for step in steplst:
                tag = '%s_min_%s_max_%s_step_%s_center_%s_class_%s'%(dataset_name,minimum,maximum,step,center,class_num)
                outdir = '%s/%s/%s' % (outpath, center, tag)
                if not os.path.exists('%s/qsublog' % outdir):
                    os.system('mkdir -p %s/qsublog' % outdir)
                walltime = 'walltime=24:00:00'
                errfile = '%s/qsublog/err' % outdir
                outfile = '%s/qsublog/out' % outdir
                run_CalmCSM = '%s/run_CalmCSM.sh' % (outdir)

                g = open(run_CalmCSM, 'w+')
                g.writelines('#!/usr/bin/bash\n')
                g.writelines('dataset_name=\"%s\"\n' % dataset_name)
                g.writelines('echo $dataset_name\n')
                g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
                g.writelines('%s %s -o %s --min %s --max %s --step %s --center %s --class_num %s\n' % (app, dataset_name, outdir, minimum,maximum,step,center,class_num))
                g.writelines("echo 'end at:' `date`\n")
                g.close()
                os.system('chmod 755 %s' % run_CalmCSM)
                os.system('/public/home/sry/bin/getQ.pl')
                os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, tag, run_CalmCSM))
                time.sleep(0.1)

