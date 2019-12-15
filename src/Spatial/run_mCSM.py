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

class_numlst = [2,8]
centerlst = ['CA','geometric']
minimum = 0.1
maximunlst = [5,6,7,8,9,10,11,12,13,14,15]
# maximunlst = [14]
steplst    = [0.5, 1, 1.5, 2]

# for wild structure
for class_num in class_numlst:
    for center in centerlst:
        for maximum in maximunlst:
            for step in steplst:
                tag = 'wild_min_%.1f_max_%.1f_step_%.1f_center_%s_class_%s'%(minimum,maximum,step,center,class_num)
                outdir = '%s/%s/%s' % (wild_outpath, center, tag)
                if not os.path.exists('%s/qsublog' % outdir):
                    os.system('mkdir -p %s/qsublog' % outdir)
                walltime = 'walltime=240:00:00'
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

# for mutant_structure