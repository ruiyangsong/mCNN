#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Revoked by CalMSA.py
'''
import os, time

def qsub_wild(app, libdir, seq_path, qsubid_prefix):
    pdbid_dir_lst = [seq_path+'/'+x for x in os.listdir(seq_path)]
    for pdbdir in pdbid_dir_lst:
        pdbid = pdbdir.split('/')[-1][0:4]
        for chain_name in os.listdir(pdbdir):
            outdir = '%s/%s'%(pdbdir, chain_name)
            qsubid = '%s_%s_%s'%(qsubid_prefix,pdbid,chain_name)
            file = '%s/%s_%s.fasta'%(outdir,pdbid,chain_name)
            if not os.path.exists('%s/qsub_log'%outdir):
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
            os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, qsubid, run_psiblast))
            time.sleep(1)

def qsub_mutant(app, libdir, seq_path, qsubid_prefix):
    tag_dir_lst = [seq_path+'/'+x for x in os.listdir(seq_path)]
    for tagdir in tag_dir_lst:
        tag = tagdir.split('/')[-1]
        pdbid,wtaa,chain_name = tag.split('_')[:3]
        outdir = tagdir
        qsubid = '%s_%s'%(qsubid_prefix,tag)
        file = '%s/%s_%s.fasta'%(outdir,pdbid,chain_name)
        if not os.path.exists('%s/qsub_log'%outdir):
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
        os.system('qsub -e %s -o %s -l %s -N %s %s' % (errfile, outfile, walltime, qsubid, run_psiblast))
        time.sleep(1)
if __name__ == '__main__':
    pass
