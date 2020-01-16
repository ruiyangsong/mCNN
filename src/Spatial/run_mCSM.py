#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time
from mCNN.processing import check_qsub

def main():
    dataset_name = sys.argv[1]

    app = '/public/home/sry/mCNN/src/Spatial/mCSM.py'
    os.system('chmod 755 %s' % app)

    wild_outpath   = '/public/home/sry/mCNN/dataset/%s/feature/mCSM/wild'%dataset_name
    mutant_outpath = '/public/home/sry/mCNN/dataset/%s/feature/mCSM/mutant'%dataset_name
    if not os.path.exists(wild_outpath):
        os.makedirs(wild_outpath)
    if not os.path.exists(mutant_outpath):
        os.makedirs(mutant_outpath)

    # class_numlst = [2,8]
    # centerlst = ['CA','geometric']
    # minimum = 0.1
    # maximunlst = [5,6,7,8,9,10,11,12,13,14,15]
    # steplst    = [0.5, 1, 1.5, 2]


    class_numlst = [2,8]# @@++
    centerlst    = ['CA','geometric']# @@++
    minimum      = 0.5# @@++
    maximumlst   = [5]# @@++
    steplst      = [1.5,2]# @@++

    wild_mCSM(dataset_name,app,wild_outpath,class_numlst,centerlst,minimum,maximumlst,steplst)
    mutant_mCSM(dataset_name,app,mutant_outpath,class_numlst,centerlst,minimum,maximumlst,steplst)
    check_qsub(tag = 'mCSM', sleep_time = 5)

def wild_mCSM(dataset_name,app,wild_outpath,class_numlst,centerlst,minimum,maximumlst,steplst):
    # for wild structure
    wild_csv_feature_dir = '/public/home/sry/mCNN/dataset/%s/feature/mCNN/wild/csv'%dataset_name

    for class_num in class_numlst:
        for center in centerlst:
            for maximum in maximumlst:
                for step in steplst:
                    qsubid = 'mCSM_wild_%s_min_%.1f_max_%.1f_step_%.1f_center_%s_class_%s'%(dataset_name,minimum,maximum,step,center,class_num)
                    qsubtag = 'min_%.1f_max_%.1f_step_%.1f_center_%s_class_%s'%(minimum,maximum,step,center,class_num)
                    outdir = '%s/npz' % wild_outpath
                    qsuboutdir = '%s/qsublog/%s' % (wild_outpath,qsubtag)
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)
                    if not os.path.exists(qsuboutdir):
                        os.system('mkdir -p %s' % qsuboutdir)
                    walltime = 'walltime=240:00:00'
                    errfile = '%s/err' % qsuboutdir
                    outfile = '%s/out' % qsuboutdir
                    run_CalmCSM = '%s/run_CalmCSM.sh' % qsuboutdir

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

def mutant_mCSM(dataset_name,app,mutant_outpath,class_numlst,centerlst,minimum,maximumlst,steplst):
    # for mutant_structure
    mutant_csv_feature_dir = '/public/home/sry/mCNN/dataset/%s/feature/mCNN/mutant/csv'%dataset_name

    for class_num in class_numlst:
        for center in centerlst:
            for maximum in maximumlst:
                for step in steplst:
                    qsubid = 'mCSM_mutant_%s_min_%.1f_max_%.1f_step_%.1f_center_%s_class_%s'%(dataset_name,minimum,maximum,step,center,class_num)
                    qsubtag = 'min_%.1f_max_%.1f_step_%.1f_center_%s_class_%s'%(minimum,maximum,step,center,class_num)
                    outdir = '%s/npz' % mutant_outpath
                    qsuboutdir = '%s/qsublog/%s'%(mutant_outpath,qsubtag)
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)
                    if not os.path.exists(qsuboutdir):
                        os.system('mkdir -p %s' % qsuboutdir)
                    walltime = 'walltime=240:00:00'
                    errfile = '%s/err' % qsuboutdir
                    outfile = '%s/out' % qsuboutdir
                    run_CalmCSM = '%s/run_CalmCSM.sh' %qsuboutdir

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

if __name__ == '__main__':
    main()