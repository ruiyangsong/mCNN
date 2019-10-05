#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, time

def shell(cmd):
    res=os.popen(cmd).readlines()[0].strip()
    return res

# dataset_name = sys.argv[1]
dataset_name = 'S2648'
user = 'sry'
app = '/public/application/ncbi-blast-2.3.0+/bin/psiblast'
#libdir = '/public/library/uniprot20'
libdir = '/library/nr/nr'
datadir = '/public/home/sry/mCNN/datasets/%s/seq%s'%(dataset_name,dataset_name)

file_lst= os.listdir(datadir)
#sub_lst = file_lst[0:2]
for i in range(len(file_lst)):
    file=file_lst[i]
    filename = file[:-6] # without extensions.
    file = '%s/%s'%(datadir,file)
    #print(file)
    tag = 'psiblast_%s'%(filename)
    #tmpdir = '/tmp/sry/%s'%tag
    outdir = '/public/home/sry/mCNN/msa/psiblast%s/%s'%(dataset_name,tag)
    os.system('mkdir -p %s/qsublog'%(outdir))
    #os.system('cp %s/%s %s'%(datadir,file,tmpdir))
    os.chdir(outdir)
    walltime = 'walltime=24:00:00'
    errfile = './qsublog/err'
    outfile = './qsublog/out'
    run_psiblast = 'run_psiblast_%s.sh'%filename
    blast_out = '%s/blast.out'%outdir
    os.system('touch %s'%blast_out)
    g = open(run_psiblast, 'w+')
    g.writelines('#!/usr/bin/bash\n')
    #g.writelines('dataset_nam=\"%s\"\n'%dataset_name)
    g.writelines('blast_out=\"%s\"\n'%blast_out)
    g.writelines('echo $blast_out\n')
    #g.writelines('`touch ${blast.out}`\n')
    g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
    g.writelines('%s -query %s -db %s -out $blast_out -num_iterations 3\n'%(app,file,libdir))
    g.writelines("echo 'end at:' `date`\n")
    #g.writelines('mv . ${outdir}')
    g.close()

    os.system('chmod 755 %s' % run_psiblast)
    os.system('/public/home/sry/bin/getQ.pl')
    os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, tag, run_psiblast))
    #print(shell('hostname'))
    #print(os.getcwd())
    #print('qsub -e %s -o %s -l %s -N %s %s' %(errfile,outfile, walltime,tag,run_psiblast))
    #print(os.getcwd())
    #print('%s submitted.'%file)
    #print(shell('hostname'))
    time.sleep(1)
