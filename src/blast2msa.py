#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os,sys
# dataset_name = sys.argv[1]
dataset_name = 'S2648'
datadir = '../msa/psiblast%s'%dataset_name #pdbS2648
blastname_lst = [x for x in os.listdir(datadir) if os.path.isdir('%s/%s'%(datadir,x))]
for blastname in blastname_lst:
    seqname = blastname[9:]
    seq = '../datasets/%s/seq%s/%s.fasta'%(dataset_name,dataset_name,seqname)
    # print('cp %s %s/%s/'%(seq,datadir,blastname))
    os.system('cp %s %s/%s/'%(seq,datadir,blastname))
    blast_out = '%s/%s/blast.out'%(datadir,blastname)
    msa_aln = '%s/%s/msa.aln'%(datadir,blastname)
    print(blast_out,msa_aln,seq)
    os.system('./alignblast.pl %s %s -Q %s -psi'%(blast_out,msa_aln,seq))