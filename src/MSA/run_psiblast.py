#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, time

def qsub(app, libdir, datapath):
    for flag in ['wild','mutant']:
        flag_path = '%s/%s'%(datapath,flag)
        for pdbid in os.listdir(flag_path):
            pdbid_path = '%s/%s'%(flag_path,pdbid)
            for chain_name in os.listdir(pdbid_path):
                outdir = '%s/%s'%(pdbid_path,chain_name)
                tag = '%s_%s_%s'%(flag,pdbid,chain_name)
                file = '%s/%s_%s.fasta'%(outdir,pdbid,chain_name)
                os.makedirs('%s/qsub_log'%outdir)
                os.chdir(outdir)
                walltime = 'walltime=240:00:00'
                errfile = './qsublog/err'
                outfile = './qsublog/out'
                run_psiblast = 'run_prog.sh'
                blast_out = '%s/blast.out'%outdir
                os.system('touch %s'%blast_out)
                g = open(run_psiblast, 'w')
                g.writelines('#!/usr/bin/bash\n')
                g.writelines('blast_out=\"%s\"\n'%blast_out)
                g.writelines('echo $blast_out\n')
                g.writelines("echo 'user:' `whoami`\necho 'hostname:' `hostname`\necho 'begin at:' `date`\n")
                g.writelines('%s -query %s -db %s -out $blast_out -num_iterations 3\n'%(app,file,libdir))
                g.writelines("echo 'end at:' `date`\n")
                g.close()

                os.system('chmod 755 %s' % run_psiblast)
                os.system('/public/home/sry/bin/getQ.pl')
                os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, tag, run_psiblast))
                time.sleep(1)

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    app = '/public/application/ncbi-blast-2.3.0+/bin/psiblast'
    libdir = '/library/nr/nr'
