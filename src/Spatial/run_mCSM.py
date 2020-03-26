#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time
from mCNN.processing import shell, check_qsub, log

def main():
    dataset_name = sys.argv[1]
    homedir = shell('echo $HOME')

    if homedir == '/home/sry':
        print('---On server ibm')
        maximumlst = [11, 12, 13, 14, 15]
    elif homedir == '/public/home/sry':
        print('---On server hp')
        maximumlst = [5,6,7,8,9,10]
    else:
        print('[ERROR] Failed to select server!')
        exit(1)


    class_numlst = [2,8]
    centerlst = ['CA','geometric']
    minimum = 0.1
    # maximumlst = [5,6,7,8,9,10,11,12,13,14,15]
    steplst    = [0.5, 1, 1.5, 2]


    # class_numlst = [2,8]# @@++
    # centerlst    = ['CA','geometric']# @@++
    # minimum      = 0.5# @@++
    # maximumlst   = [5]# @@++
    # steplst      = [1.5]# @@++

    QR = QsubRunner(homedir,dataset_name,class_numlst,centerlst,minimum,maximumlst,steplst)
    QR.mCSM_runner()

class QsubRunner(object):
    def __init__(self,homedir,dataset_name,class_numlst,centerlst,minimum,maximumlst,steplst):
        self.homedir       = homedir
        self.dataset_name  = dataset_name
        self.sleep_time    = 5

        self.app = '%s/mCNN/src/Spatial/mCSM.py'%self.homedir

        self.class_numlst = class_numlst
        self.centerlst    = centerlst
        self.minimum      = minimum
        self.maximumlst   = maximumlst
        self.steplst      = steplst

        self.wild_outpath   = '%s/mCNN/dataset/%s/feature/mCSM/wild' % (self.homedir,self.dataset_name)
        self.mutant_outpath = '%s/mCNN/dataset/%s/feature/mCSM/mutant' % (self.homedir,self.dataset_name)
        if not os.path.exists(self.wild_outpath):
            os.makedirs(self.wild_outpath)
        if not os.path.exists(self.mutant_outpath):
            os.makedirs(self.mutant_outpath)

    @log
    def mCSM_runner(self):
        for class_num in self.class_numlst:
            for center in self.centerlst:
                for maximum in self.maximumlst:
                    for step in self.steplst:
                        self.run_wild_mCSM(class_num,center,maximum,step)
                        self.run_mutant_mCSM(class_num,center,maximum,step)
        check_qsub(tag='mCSM_%s'%self.dataset_name, sleep_time=self.sleep_time)

    def run_wild_mCSM(self,class_num,center,maximum,step):
        # for wild structure
        wild_csv_feature_dir = '%s/mCNN/dataset/%s/feature/mCNN/wild/csv'%(self.homedir,self.dataset_name)
        qsubid = 'mCSM_%s_wild_min_%.1f_max_%.1f_step_%.1f_center_%s_class_%s'%(self.dataset_name,self.minimum,maximum,step,center,class_num)
        qsubtag = 'min_%.1f_max_%.1f_step_%.1f_center_%s_class_%s'%(self.minimum,maximum,step,center,class_num)
        outdir = '%s/npz' % self.wild_outpath
        qsuboutdir = '%s/qsublog/%s' % (self.wild_outpath,qsubtag)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if not os.path.exists(qsuboutdir):
            os.system('mkdir -p %s' % qsuboutdir)
        walltime = 'walltime=240:00:00'
        errfile = '%s/err' % qsuboutdir
        outfile = '%s/out' % qsuboutdir
        run_CalmCSM = '%s/run_CalmCSM.sh' % qsuboutdir

        g = open(run_CalmCSM, 'w+')
        g.writelines('#!/usr/bin/env bash\n')
        g.writelines('dataset_name=\"%s\"\n' % self.dataset_name)
        g.writelines('echo $dataset_name\n')
        g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
        g.writelines('%s %s %s -o %s --min %s --max %s --step %s --center %s --class_num %s\n' % (self.app, self.dataset_name, wild_csv_feature_dir, outdir, self.minimum,maximum,step,center,class_num))
        g.writelines("echo 'end at:' `date`\n")
        g.close()
        os.system('chmod 755 %s' % run_CalmCSM)
        os.system('%s/bin/getQ.pl'%self.homedir)
        os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, qsubid, run_CalmCSM))
        time.sleep(0.01)

    def run_mutant_mCSM(self,class_num,center,maximum,step):
        # for mutant_structure
        mutant_csv_feature_dir = '%s/mCNN/dataset/%s/feature/mCNN/mutant/csv'%(self.homedir,self.dataset_name)
        qsubid = 'mCSM_%s_mutant_min_%.1f_max_%.1f_step_%.1f_center_%s_class_%s'%(self.dataset_name,self.minimum,maximum,step,center,class_num)
        qsubtag = 'min_%.1f_max_%.1f_step_%.1f_center_%s_class_%s'%(self.minimum,maximum,step,center,class_num)
        outdir = '%s/npz' % self.mutant_outpath
        qsuboutdir = '%s/qsublog/%s'%(self.mutant_outpath,qsubtag)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if not os.path.exists(qsuboutdir):
            os.system('mkdir -p %s' % qsuboutdir)
        walltime = 'walltime=240:00:00'
        errfile = '%s/err' % qsuboutdir
        outfile = '%s/out' % qsuboutdir
        run_CalmCSM = '%s/run_CalmCSM.sh' %qsuboutdir

        g = open(run_CalmCSM, 'w+')
        g.writelines('#!/usr/bin/env bash\n')
        g.writelines('dataset_name=\"%s\"\n' % self.dataset_name)
        g.writelines('echo $dataset_name\n')
        g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
        g.writelines('%s %s %s -o %s --min %s --max %s --step %s --center %s --class_num %s\n' % (self.app, self.dataset_name, mutant_csv_feature_dir, outdir, self.minimum,maximum,step,center,class_num))
        g.writelines("echo 'end at:' `date`\n")
        g.close()
        os.system('chmod 755 %s' % run_CalmCSM)
        os.system('%s/bin/getQ.pl'%self.homedir)
        os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, qsubid, run_CalmCSM))
        time.sleep(0.01)

if __name__ == '__main__':
    main()