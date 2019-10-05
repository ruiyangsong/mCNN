#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---gengrete data array on cluster---
import os, time

def shell(cmd):
    res=os.popen(cmd).readlines()[0].strip()
    return res

user = 'sry'
# app = '/public/application/ncbi-blast-2.3.0+/bin/psiblast'

#f = open('../shell/from_caobx/calc_all.sh')
f = open('test.py')
lines = f.readlines()
f.close()
for line in lines:
    lst = line.split(' ')
    tag = lst[2] + '_' + lst[3] + '_' + lst[4]
    print(tag)
    # outdir = '/public/home/sry/mCNN/datasets_array/%s'%tag
    outdir = '../datasets_array/%s' % tag
    os.system('mkdir -p %s'%(outdir))
    walltime = 'walltime=24:00:00'
    errfile = '%s/qsub.err'%outdir
    outfile = '%s/qsub.out'%outdir
    run_calc = '%s/run_calc_%s.sh'%(outdir,tag)
    g = open(run_calc, 'w+')
    g.writelines('#!/usr/bin/bash\n')
    g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
    g.writelines("echo 'path:' `pwd`\n")
    g.writelines(line)
    g.writelines("echo 'end at:' `date`\n")
    g.close()
    os.system('chmod 755 %s' % run_calc)
    os.system('/public/home/sry/bin/getQ.pl')
    os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, tag, run_calc))
    #print(shell('hostname'))
    #print(os.getcwd())
    #print('qsub -e %s -o %s -l %s -N %s %s' %(errfile,outfile, walltime,tag,run_psiblast))
    #print(os.getcwd())
    #print('%s submitted.'%file)
    #print(shell('hostname'))
    time.sleep(1)
